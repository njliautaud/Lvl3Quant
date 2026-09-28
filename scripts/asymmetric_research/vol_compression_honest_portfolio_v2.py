#!/usr/bin/env python3
"""
Vol Compression Breakout — HONEST Portfolio Sim v2

The v1 portfolio sim showed 597% CAGR which is suspiciously high.
This version does proper daily mark-to-market with:
- Realistic transaction costs ($0.005/share + 0.05% slippage)
- Position sizing (equal weight, max 10 positions)
- Proper cash management (no leverage)
- Regime gap test (bull vs bear daily Sharpe)
- Permutation test (random entry dates)

HC #741: Must report CAGR, MaxDD, Sharpe, Sortino, WR
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json, os, sys, warnings, time
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/nick/Lvl3Quant/output/vol_compression_honest_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)

def get_sp500_tickers():
    cache = '/home/nick/Lvl3Quant/output/sp500_tickers.json'
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    tickers = sorted(tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist())
    with open(cache, 'w') as f:
        json.dump(tickers, f)
    return tickers

def download_data(tickers, start='2010-01-01', end='2026-07-22'):
    """Download price data in batches."""
    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        log(f"  Downloading batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1} ({len(batch)} tickers)")
        try:
            data = yf.download(batch, start=start, end=end, progress=False, auto_adjust=True, threads=True)
            if data is None or data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        if t in data.columns.get_level_values(1):
                            td = data.xs(t, level=1, axis=1)
                        elif t in data.columns.get_level_values(0):
                            td = data[t]
                        else:
                            continue
                        if 'Close' in td.columns and len(td.dropna(subset=['Close'])) > 100:
                            all_data[t] = td[['Open','High','Low','Close','Volume']].dropna()
                    except:
                        pass
            else:
                if len(batch) == 1 and 'Close' in data.columns:
                    all_data[batch[0]] = data[['Open','High','Low','Close','Volume']].dropna()
        except Exception as e:
            log(f"  Batch error: {e}")
        time.sleep(0.5)
    return all_data

def detect_breakouts(prices_dict, vol_percentile=10, lookback=63):
    """Detect vol compression breakouts across all stocks."""
    signals = []
    
    for ticker, df in prices_dict.items():
        if len(df) < lookback + 20:
            continue
        
        df = df.copy()
        df['ret'] = df['Close'].pct_change()
        df['vol_20d'] = df['ret'].rolling(20).std()
        
        # Rolling percentile of volatility
        df['vol_pctile'] = df['vol_20d'].rolling(lookback).apply(
            lambda x: (x < x.iloc[-1]).sum() / len(x) * 100, raw=False
        )
        
        # Breakout = first day vol pctile goes from <10 to >10 (compression ended)
        df['compressed'] = df['vol_pctile'] < vol_percentile
        df['breakout'] = df['compressed'].shift(1) & ~df['compressed']
        df['direction'] = np.where(df['ret'] > 0, 'long', 'short')
        
        breakout_days = df[df['breakout']].index
        for day in breakout_days:
            idx = df.index.get_loc(day)
            if idx + 10 >= len(df):
                continue
            
            entry_price = df.iloc[idx]['Close']
            direction = df.iloc[idx]['direction']
            
            # Get exit prices at various horizons
            exit_5d = df.iloc[min(idx+5, len(df)-1)]['Close'] if idx+5 < len(df) else None
            exit_10d = df.iloc[min(idx+10, len(df)-1)]['Close'] if idx+10 < len(df) else None
            
            signals.append({
                'ticker': ticker,
                'date': day,
                'entry_price': entry_price,
                'direction': direction,
                'vol_pctile': df.iloc[idx-1]['vol_pctile'] if idx > 0 else None,
                'exit_5d': exit_5d,
                'exit_10d': exit_10d,
            })
    
    return pd.DataFrame(signals)

def run_portfolio_sim(signals_df, prices_dict, max_positions=10, hold_days=10, 
                       initial_capital=100000, cost_per_share=0.005, slippage_pct=0.0005):
    """Run proper daily MTM portfolio simulation."""
    
    # Sort signals by date
    signals_df = signals_df.sort_values('date').copy()
    
    # Get SPY for regime classification
    spy_data = prices_dict.get('SPY', None)
    if spy_data is None:
        log("WARNING: No SPY data for regime classification")
        return None
    
    spy_data = spy_data.copy()
    spy_data['sma200'] = spy_data['Close'].rolling(200).mean()
    spy_data['regime'] = np.where(spy_data['Close'] > spy_data['sma200'], 'bull', 'bear')
    
    # Build daily P&L series
    all_dates = sorted(set().union(*[set(df.index) for df in prices_dict.values()]))
    all_dates = [d for d in all_dates if d >= signals_df['date'].min()]
    
    portfolio = {
        'cash': initial_capital,
        'positions': [],  # list of {ticker, entry_price, entry_date, direction, shares, exit_date}
        'nav_history': [],
        'trades': [],
    }
    
    signals_by_date = signals_df.groupby('date')
    
    for date in all_dates:
        # 1. Check for exits (positions past hold period)
        new_positions = []
        for pos in portfolio['positions']:
            days_held = len([d for d in all_dates if pos['entry_date'] <= d <= date])
            if days_held >= hold_days:
                # Exit
                if pos['ticker'] in prices_dict and date in prices_dict[pos['ticker']].index:
                    exit_price = prices_dict[pos['ticker']].loc[date, 'Close']
                else:
                    exit_price = pos.get('last_price', pos['entry_price'])
                
                # Calculate P&L
                if pos['direction'] == 'long':
                    pnl = (exit_price - pos['entry_price']) * pos['shares']
                else:
                    pnl = (pos['entry_price'] - exit_price) * pos['shares']
                
                # Subtract costs
                cost = cost_per_share * pos['shares'] * 2 + exit_price * pos['shares'] * slippage_pct
                pnl -= cost
                
                portfolio['cash'] += exit_price * pos['shares'] + pnl
                portfolio['trades'].append({
                    'ticker': pos['ticker'],
                    'direction': pos['direction'],
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'return_pct': pnl / (pos['entry_price'] * pos['shares']) * 100,
                })
            else:
                new_positions.append(pos)
        portfolio['positions'] = new_positions
        
        # 2. Check for new entries
        if date in signals_by_date.groups and len(portfolio['positions']) < max_positions:
            day_signals = signals_by_date.get_group(date)
            slots = max_positions - len(portfolio['positions'])
            
            for _, sig in day_signals.head(slots).iterrows():
                if sig['ticker'] not in prices_dict:
                    continue
                if date not in prices_dict[sig['ticker']].index:
                    continue
                    
                entry_price = sig['entry_price']
                # Position size = equal weight
                pos_value = portfolio['cash'] / max(slots, 1)
                pos_value = min(pos_value, portfolio['cash'] * 0.95)  # Keep 5% cash buffer
                if pos_value < 100 or entry_price < 1:
                    continue
                
                shares = int(pos_value / entry_price)
                if shares < 1:
                    continue
                
                cost = cost_per_share * shares + entry_price * shares * slippage_pct
                portfolio['cash'] -= entry_price * shares + cost
                
                portfolio['positions'].append({
                    'ticker': sig['ticker'],
                    'entry_price': entry_price,
                    'entry_date': date,
                    'direction': sig['direction'],
                    'shares': shares,
                })
        
        # 3. Mark to market
        position_value = 0
        for pos in portfolio['positions']:
            if pos['ticker'] in prices_dict and date in prices_dict[pos['ticker']].index:
                current_price = prices_dict[pos['ticker']].loc[date, 'Close']
                pos['last_price'] = current_price
                if pos['direction'] == 'long':
                    position_value += current_price * pos['shares']
                else:
                    # Short: value = 2*entry - current (profit when price drops)
                    position_value += (2 * pos['entry_price'] - current_price) * pos['shares']
            else:
                position_value += pos.get('last_price', pos['entry_price']) * pos['shares']
        
        nav = portfolio['cash'] + position_value
        regime = spy_data.loc[date, 'regime'] if date in spy_data.index else 'unknown'
        
        portfolio['nav_history'].append({
            'date': date,
            'nav': nav,
            'cash': portfolio['cash'],
            'positions': len(portfolio['positions']),
            'regime': regime,
        })
    
    return portfolio

def analyze_results(portfolio, initial_capital=100000):
    """Compute all metrics including regime gap."""
    nav_df = pd.DataFrame(portfolio['nav_history'])
    nav_df['date'] = pd.to_datetime(nav_df['date'])
    nav_df = nav_df.set_index('date')
    nav_df['daily_ret'] = nav_df['nav'].pct_change()
    
    # Overall metrics
    total_days = len(nav_df)
    years = total_days / 252
    final_nav = nav_df['nav'].iloc[-1]
    cagr = (final_nav / initial_capital) ** (1/years) - 1
    
    daily_rets = nav_df['daily_ret'].dropna()
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    
    neg_rets = daily_rets[daily_rets < 0]
    sortino = daily_rets.mean() / neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 and neg_rets.std() > 0 else 0
    
    # Max drawdown
    cum_max = nav_df['nav'].cummax()
    drawdown = (nav_df['nav'] - cum_max) / cum_max
    max_dd = drawdown.min()
    
    # Trade stats
    trades = portfolio['trades']
    if trades:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        wr = len(wins) / len(trades) * 100
        pf = abs(sum(t['pnl'] for t in wins)) / abs(sum(t['pnl'] for t in losses)) if losses else float('inf')
    else:
        wr, pf = 0, 0
    
    # Regime gap
    bull_rets = nav_df[nav_df['regime'] == 'bull']['daily_ret'].dropna()
    bear_rets = nav_df[nav_df['regime'] == 'bear']['daily_ret'].dropna()
    
    bull_sharpe = bull_rets.mean() / bull_rets.std() * np.sqrt(252) if len(bull_rets) > 10 and bull_rets.std() > 0 else 0
    bear_sharpe = bear_rets.mean() / bear_rets.std() * np.sqrt(252) if len(bear_rets) > 10 and bear_rets.std() > 0 else 0
    
    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    
    # Per-year breakdown
    nav_df['year'] = nav_df.index.year
    yearly = []
    for yr, grp in nav_df.groupby('year'):
        yr_ret = grp['nav'].iloc[-1] / grp['nav'].iloc[0] - 1
        yearly.append({'year': yr, 'return': yr_ret * 100})
    
    results = {
        'cagr': cagr * 100,
        'max_dd': max_dd * 100,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': wr,
        'profit_factor': pf,
        'total_trades': len(trades),
        'bull_sharpe': bull_sharpe,
        'bear_sharpe': bear_sharpe,
        'regime_gap': regime_gap,
        'final_nav': final_nav,
        'years': years,
        'yearly': yearly,
    }
    
    return results, nav_df

def run_permutation_test(signals_df, prices_dict, real_sharpe, n_perms=200, **sim_kwargs):
    """Permutation test: random entry dates."""
    log(f"Running {n_perms} permutation shuffles...")
    
    all_dates = sorted(set().union(*[set(df.index) for df in prices_dict.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp('2011-01-01')]
    
    perm_sharpes = []
    for i in range(n_perms):
        # Shuffle entry dates (keep everything else)
        shuffled = signals_df.copy()
        shuffled['date'] = np.random.choice(all_dates, size=len(shuffled), replace=True)
        shuffled = shuffled.sort_values('date')
        
        try:
            perm_portfolio = run_portfolio_sim(shuffled, prices_dict, **sim_kwargs)
            if perm_portfolio:
                nav_df = pd.DataFrame(perm_portfolio['nav_history'])
                daily_rets = nav_df['nav'].pct_change().dropna()
                perm_sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
                perm_sharpes.append(perm_sharpe)
        except:
            pass
        
        if (i+1) % 50 == 0:
            log(f"  Permutation {i+1}/{n_perms} done")
    
    if perm_sharpes:
        p_value = sum(1 for s in perm_sharpes if s >= real_sharpe) / len(perm_sharpes)
        return p_value, np.mean(perm_sharpes), np.std(perm_sharpes)
    return 1.0, 0, 0

def main():
    log("=" * 60)
    log("VOL COMPRESSION BREAKOUT — HONEST PORTFOLIO v2")
    log("=" * 60)
    
    # Get tickers
    tickers = get_sp500_tickers()
    # Add SPY for regime
    if 'SPY' not in tickers:
        tickers.append('SPY')
    
    log(f"Downloading data for {len(tickers)} tickers...")
    prices_dict = download_data(tickers, start='2010-01-01', end='2026-07-22')
    log(f"Got data for {len(prices_dict)} tickers")
    
    # Detect breakout signals
    log("Detecting vol compression breakouts...")
    signals = detect_breakouts(prices_dict, vol_percentile=10, lookback=63)
    log(f"Found {len(signals)} breakout signals")
    
    # Run portfolio sim for different hold periods
    for hold_days in [5, 10]:
        log(f"\n{'='*40}")
        log(f"PORTFOLIO SIM — {hold_days}d hold, max 10 positions")
        log(f"{'='*40}")
        
        portfolio = run_portfolio_sim(signals, prices_dict, max_positions=10, hold_days=hold_days)
        if not portfolio:
            log("FAILED — no portfolio generated")
            continue
        
        results, nav_df = analyze_results(portfolio)
        
        log(f"\n  RESULTS:")
        log(f"  CAGR: {results['cagr']:.1f}%")
        log(f"  MaxDD: {results['max_dd']:.1f}%")
        log(f"  Sharpe: {results['sharpe']:.3f}")
        log(f"  Sortino: {results['sortino']:.3f}")
        log(f"  Win Rate: {results['win_rate']:.1f}%")
        log(f"  Profit Factor: {results['profit_factor']:.2f}")
        log(f"  Trades: {results['total_trades']}")
        log(f"  Bull Sharpe: {results['bull_sharpe']:.3f}")
        log(f"  Bear Sharpe: {results['bear_sharpe']:.3f}")
        log(f"  Regime Gap: {results['regime_gap']:.3f} {'PASS' if results['regime_gap'] < 0.50 else 'FAIL'}")
        log(f"  Final NAV: ${results['final_nav']:,.0f} (from $100K)")
        
        log(f"\n  YEARLY:")
        for yr in results['yearly']:
            marker = '🟢' if yr['return'] > 0 else '🔴'
            log(f"    {yr['year']}: {marker} {yr['return']:+.1f}%")
        
        profitable_years = sum(1 for yr in results['yearly'] if yr['return'] > 0)
        log(f"  {profitable_years}/{len(results['yearly'])} years profitable")
        
        # Permutation test
        perm_p, perm_mean, perm_std = run_permutation_test(
            signals, prices_dict, results['sharpe'], 
            n_perms=200, max_positions=10, hold_days=hold_days
        )
        log(f"\n  PERMUTATION TEST:")
        log(f"  p-value: {perm_p:.3f} {'PASS' if perm_p < 0.05 else 'FAIL'}")
        log(f"  Null mean Sharpe: {perm_mean:.3f} ± {perm_std:.3f}")
        log(f"  Observed Sharpe: {results['sharpe']:.3f}")
        
        # Save results
        results['perm_p'] = perm_p
        results['perm_mean_sharpe'] = perm_mean
        results['hold_days'] = hold_days
        with open(f'{OUTPUT_DIR}/results_{hold_days}d.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        nav_df.to_csv(f'{OUTPUT_DIR}/nav_{hold_days}d.csv')
    
    log("\n" + "=" * 60)
    log("COMPLETE")
    log("=" * 60)

if __name__ == '__main__':
    main()
