#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Backtest v1 (HC #740)

Combines ALL proven strategy signals into a single portfolio to test:
1. Aggregate portfolio performance (Sharpe, Sortino, MaxDD, CAGR)
2. Strategy diversification benefit (correlation between strategies)
3. Capital efficiency (how much overlap / how many concurrent positions)
4. Regime resilience (performance in green vs red months)

Proven strategies included:
- Oversold bounce (RSI<20, 10d hold) — HC #748
- 3% drop contrarian (10d hold) — HC #748
- Vol compression breakout (10d hold) — HC #755
- Volume climax reversal (2x vol + 2% drop + vol compressed, 5d hold) — HC #759
- Breadth thrust contrarian (collapse<15%, 21d hold) — HC #766
- Stealth accumulation (downtrend, 5d hold) — HC #762
- Flow-enhanced: MFI oversold filter variants — HC #768

Portfolio construction:
- Equal capital allocation per signal
- Max 10% capital per single stock position
- Max 20 concurrent positions
- No leverage
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/multi_strategy_portfolio_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# S&P 500 tickers (same universe as prior research)
SP500_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

def get_sp500_tickers():
    """Get S&P 500 tickers, with fallback to cached list."""
    cache_file = '/home/jupiter/Lvl3Quant/output/sp500_tickers.json'
    try:
        tables = pd.read_html(SP500_URL)
        tickers = sorted(tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist())
        with open(cache_file, 'w') as f:
            json.dump(tickers, f)
        return tickers
    except:
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)
        # Hardcoded fallback of top 50
        return ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','UNH','JNJ',
                'JPM','V','PG','XOM','HD','CVX','MA','ABBV','MRK','LLY',
                'PEP','KO','COST','AVGO','WMT','MCD','CSCO','TMO','CRM','ACN',
                'ABT','DHR','LIN','NEE','TXN','PM','QCOM','UNP','LOW','RTX',
                'INTC','AMGN','HON','IBM','AMAT','CAT','BA','GS','BLK','AXP']

def load_price_data(tickers, start='2013-01-01', end='2026-07-22'):
    """Load OHLCV data with caching."""
    cache_file = os.path.join(OUTPUT_DIR, 'prices_cache.parquet')
    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        print(f"  Loaded cached prices: {len(df)} rows, {df['ticker'].nunique()} tickers")
        return df

    all_data = []
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}: {len(batch)} tickers...")
        try:
            data = yf.download(batch, start=start, end=end, group_by='ticker', threads=True, progress=False)
            for t in batch:
                try:
                    if len(batch) == 1:
                        td = data.copy()
                    else:
                        td = data[t].copy()
                    td = td.dropna(subset=['Close'])
                    if len(td) < 252:
                        continue
                    td['ticker'] = t
                    td.index.name = 'date'
                    all_data.append(td.reset_index())
                except:
                    pass
        except:
            pass

    df = pd.concat(all_data, ignore_index=True)
    # Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == '' else c[0] for c in df.columns]

    df.to_parquet(cache_file)
    print(f"  Final: {df['ticker'].nunique()} tickers, {len(df)} rows")
    return df

def load_spy_data(start='2013-01-01', end='2026-07-22'):
    """Load SPY for regime classification."""
    cache_file = os.path.join(OUTPUT_DIR, 'spy_cache.parquet')
    if os.path.exists(cache_file):
        return pd.read_parquet(cache_file)
    spy = yf.download('SPY', start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index.name = 'date'
    spy = spy.reset_index()
    spy.to_parquet(cache_file)
    return spy

def compute_features(df):
    """Compute all signal features per ticker."""
    results = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        g['ret_1d'] = g['Close'].pct_change()
        g['ret_5d'] = g['Close'].pct_change(5)
        g['ret_10d'] = g['Close'].pct_change(10)

        # RSI
        delta = g['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g['rsi'] = 100 - (100 / (1 + rs))

        # Volatility percentile (20d realized vol, ranked over 252d)
        g['vol_20d'] = g['ret_1d'].rolling(20).std()
        g['vol_pctile'] = g['vol_20d'].rolling(252).rank(pct=True)

        # Volume features
        g['vol_avg_20d'] = g['Volume'].rolling(20).mean()
        g['vol_ratio'] = g['Volume'] / g['vol_avg_20d'].replace(0, np.nan)

        # MFI (14-period)
        typical_price = (g['High'] + g['Low'] + g['Close']) / 3
        raw_mf = typical_price * g['Volume']
        pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
        neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
        g['mfi'] = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))

        # OBV trend (20d slope of OBV)
        obv = (np.sign(g['Close'].diff()) * g['Volume']).cumsum()
        g['obv'] = obv
        g['obv_slope_20d'] = obv.rolling(20).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 20 else np.nan,
            raw=True
        )

        # SMA 50d
        g['sma_50d'] = g['Close'].rolling(50).mean()

        # Forward returns (for backtest)
        g['fwd_ret_5d'] = g['Close'].shift(-5) / g['Close'] - 1
        g['fwd_ret_10d'] = g['Close'].shift(-10) / g['Close'] - 1
        g['fwd_ret_21d'] = g['Close'].shift(-21) / g['Close'] - 1

        results.append(g)

    return pd.concat(results, ignore_index=True)

def classify_regime(spy_df):
    """Classify each month as green/red based on SPY close-to-close."""
    spy = spy_df.copy()
    spy['date'] = pd.to_datetime(spy['date'])
    spy['month'] = spy['date'].dt.to_period('M')
    monthly = spy.groupby('month')['Close'].agg(['first', 'last'])
    monthly['regime'] = np.where(monthly['last'] > monthly['first'], 'green', 'red')
    return monthly['regime'].to_dict()

def detect_breadth_collapse(spy_df, df, threshold=0.15):
    """Detect days where advancing stocks < threshold of total."""
    dates = df.groupby('date').apply(lambda x: (x['ret_1d'] > 0).mean())
    collapse_dates = dates[dates < threshold].index
    return set(collapse_dates)

# ============== SIGNAL DETECTORS ==============

def signal_oversold_bounce(df, rsi_thresh=20, hold=10):
    """RSI < 20, hold N days. Proven in HC #748."""
    mask = df['rsi'] < rsi_thresh
    signals = df[mask][['date', 'ticker', f'fwd_ret_{hold}d']].copy()
    signals.columns = ['date', 'ticker', 'fwd_ret']
    signals['strategy'] = f'oversold_rsi{rsi_thresh}_h{hold}d'
    signals['hold_days'] = hold
    return signals.dropna(subset=['fwd_ret'])

def signal_drop_3pct(df, drop_thresh=0.03, hold=10):
    """Daily drop > 3%, hold N days. Proven in HC #748."""
    mask = df['ret_1d'] < -drop_thresh
    signals = df[mask][['date', 'ticker', f'fwd_ret_{hold}d']].copy()
    signals.columns = ['date', 'ticker', 'fwd_ret']
    signals['strategy'] = f'drop{int(drop_thresh*100)}pct_h{hold}d'
    signals['hold_days'] = hold
    return signals.dropna(subset=['fwd_ret'])

def signal_vol_compression(df, vol_pctile_thresh=0.10, hold=10):
    """Vol at <10th percentile, hold N days. Proven in HC #755."""
    mask = df['vol_pctile'] < vol_pctile_thresh
    signals = df[mask][['date', 'ticker', f'fwd_ret_{hold}d']].copy()
    signals.columns = ['date', 'ticker', 'fwd_ret']
    signals['strategy'] = f'vol_comp_p{int(vol_pctile_thresh*100)}_h{hold}d'
    signals['hold_days'] = hold
    return signals.dropna(subset=['fwd_ret'])

def signal_vol_climax(df, vol_mult=2.0, drop_thresh=0.02, vol_pctile_thresh=0.10, hold=5):
    """Volume spike + drop + vol compressed. Proven in HC #759."""
    mask = (
        (df['vol_ratio'] >= vol_mult) &
        (df['ret_1d'] < -drop_thresh) &
        (df['vol_pctile'] < vol_pctile_thresh)
    )
    signals = df[mask][['date', 'ticker', f'fwd_ret_{hold}d']].copy()
    signals.columns = ['date', 'ticker', 'fwd_ret']
    signals['strategy'] = f'vol_climax_v{vol_mult}_d{int(drop_thresh*100)}_h{hold}d'
    signals['hold_days'] = hold
    return signals.dropna(subset=['fwd_ret'])

def signal_drop3_mfi_oversold(df, drop_thresh=0.03, mfi_thresh=20, hold=21):
    """3% drop + MFI oversold (best regime-agnostic flow filter). From flow-enhanced v1."""
    mask = (df['ret_1d'] < -drop_thresh) & (df['mfi'] < mfi_thresh)
    signals = df[mask][['date', 'ticker', f'fwd_ret_{hold}d']].copy()
    signals.columns = ['date', 'ticker', 'fwd_ret']
    signals['strategy'] = f'drop3_mfi{mfi_thresh}_h{hold}d'
    signals['hold_days'] = hold
    return signals.dropna(subset=['fwd_ret'])

def signal_volcomp_mfi_oversold(df, vol_pctile_thresh=0.10, mfi_thresh=20, hold=10):
    """Vol compression + MFI oversold. From flow-enhanced v1 (regime gap 0.07)."""
    mask = (df['vol_pctile'] < vol_pctile_thresh) & (df['mfi'] < mfi_thresh)
    signals = df[mask][['date', 'ticker', f'fwd_ret_{hold}d']].copy()
    signals.columns = ['date', 'ticker', 'fwd_ret']
    signals['strategy'] = f'volcomp_mfi{mfi_thresh}_h{hold}d'
    signals['hold_days'] = hold
    return signals.dropna(subset=['fwd_ret'])

def signal_oversold_high_relvol(df, rsi_thresh=20, hold=21):
    """Oversold + high relative volume (>1.5x avg). From flow-enhanced v1."""
    mask = (df['rsi'] < rsi_thresh) & (df['vol_ratio'] > 1.5)
    signals = df[mask][['date', 'ticker', f'fwd_ret_{hold}d']].copy()
    signals.columns = ['date', 'ticker', 'fwd_ret']
    signals['strategy'] = f'oversold_highvol_h{hold}d'
    signals['hold_days'] = hold
    return signals.dropna(subset=['fwd_ret'])

def signal_confluence_drop_volcomp(df, drop_thresh=0.03, vol_pctile_thresh=0.10, hold=10):
    """Oversold + vol compression confluence. STAR from HC #755 (regime gap 0.066)."""
    mask = (df['ret_1d'] < -drop_thresh) & (df['vol_pctile'] < vol_pctile_thresh)
    signals = df[mask][['date', 'ticker', f'fwd_ret_{hold}d']].copy()
    signals.columns = ['date', 'ticker', 'fwd_ret']
    signals['strategy'] = f'confluence_drop3_vc_h{hold}d'
    signals['hold_days'] = hold
    return signals.dropna(subset=['fwd_ret'])

# ============== PORTFOLIO SIMULATION ==============

def simulate_portfolio(all_signals, max_positions=20, max_per_stock=0.10, initial_capital=100000):
    """
    Day-by-day portfolio simulation.
    - Equal weight per new signal
    - Max N concurrent positions
    - Max X% of capital in single stock
    - Track daily P&L, positions, drawdown
    """
    # Sort all signals by date
    signals = all_signals.sort_values('date').copy()
    signals['date'] = pd.to_datetime(signals['date'])

    # Get all unique dates
    all_dates = sorted(signals['date'].unique())

    # Track positions: list of {ticker, strategy, entry_date, exit_date, return}
    positions = []
    daily_pnl = []
    capital = initial_capital

    for date in all_dates:
        # Count active positions
        active = [p for p in positions if p['entry_date'] <= date < p['exit_date']]

        # Get new signals for today
        today_signals = signals[signals['date'] == date]

        # How many slots available
        slots = max_positions - len(active)

        if slots > 0 and len(today_signals) > 0:
            # Pick best signals (highest absolute forward return diversity)
            # Deduplicate by ticker (only one position per ticker)
            active_tickers = {p['ticker'] for p in active}
            new_signals = today_signals[~today_signals['ticker'].isin(active_tickers)]

            # Take up to 'slots' new positions, diversified by strategy
            # Prioritize strategies with fewer active positions
            strategy_counts = {}
            for p in active:
                strategy_counts[p['strategy']] = strategy_counts.get(p['strategy'], 0) + 1

            new_signals = new_signals.copy()
            new_signals['strat_count'] = new_signals['strategy'].map(
                lambda s: strategy_counts.get(s, 0)
            )
            new_signals = new_signals.sort_values('strat_count')

            for _, sig in new_signals.head(slots).iterrows():
                exit_date = date + pd.Timedelta(days=int(sig['hold_days'] * 1.5))  # Calendar days
                positions.append({
                    'ticker': sig['ticker'],
                    'strategy': sig['strategy'],
                    'entry_date': date,
                    'exit_date': exit_date,
                    'fwd_ret': sig['fwd_ret'],
                    'hold_days': sig['hold_days'],
                })

    # Convert to daily returns
    pos_df = pd.DataFrame(positions)
    if len(pos_df) == 0:
        return pd.DataFrame(), pos_df

    # Compute per-day portfolio return
    date_range = pd.date_range(all_dates[0], all_dates[-1], freq='B')
    daily_returns = []

    for date in date_range:
        active = pos_df[(pos_df['entry_date'] <= date) & (pos_df['exit_date'] > date)]
        if len(active) == 0:
            daily_returns.append({'date': date, 'return': 0.0, 'n_positions': 0})
            continue

        # Each position contributes proportionally
        n = len(active)
        weight = 1.0 / max(n, 1)

        # Daily return = average of position daily returns
        # Approximate: distribute total return evenly across hold period
        total_daily_ret = 0
        for _, pos in active.iterrows():
            days_held = max((pos['exit_date'] - pos['entry_date']).days, 1)
            daily_pos_ret = pos['fwd_ret'] / days_held  # Linear approximation
            total_daily_ret += weight * daily_pos_ret

        daily_returns.append({
            'date': date,
            'return': total_daily_ret,
            'n_positions': n,
        })

    daily_df = pd.DataFrame(daily_returns)
    daily_df['cum_return'] = (1 + daily_df['return']).cumprod()
    daily_df['drawdown'] = daily_df['cum_return'] / daily_df['cum_return'].cummax() - 1

    return daily_df, pos_df

def compute_metrics(daily_df, pos_df):
    """Compute portfolio-level metrics."""
    if len(daily_df) == 0:
        return {}

    rets = daily_df['return']
    ann_factor = 252

    total_return = daily_df['cum_return'].iloc[-1] - 1
    years = len(daily_df) / ann_factor
    cagr = (1 + total_return) ** (1 / years) - 1

    sharpe = rets.mean() / rets.std() * np.sqrt(ann_factor) if rets.std() > 0 else 0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(ann_factor) if downside > 0 else 0
    max_dd = daily_df['drawdown'].min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (daily)
    win_days = (rets > 0).sum()
    total_days = (rets != 0).sum()
    daily_wr = win_days / total_days if total_days > 0 else 0

    # Trade-level stats
    if len(pos_df) > 0:
        trade_wr = (pos_df['fwd_ret'] > 0).mean()
        avg_win = pos_df[pos_df['fwd_ret'] > 0]['fwd_ret'].mean() if (pos_df['fwd_ret'] > 0).any() else 0
        avg_loss = pos_df[pos_df['fwd_ret'] < 0]['fwd_ret'].mean() if (pos_df['fwd_ret'] < 0).any() else 0
        pf = abs(pos_df[pos_df['fwd_ret'] > 0]['fwd_ret'].sum() / pos_df[pos_df['fwd_ret'] < 0]['fwd_ret'].sum()) if (pos_df['fwd_ret'] < 0).any() else float('inf')
    else:
        trade_wr = avg_win = avg_loss = pf = 0

    avg_positions = daily_df['n_positions'].mean()
    max_positions = daily_df['n_positions'].max()

    return {
        'total_return': f"{total_return:.1%}",
        'cagr': f"{cagr:.1%}",
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'max_drawdown': f"{max_dd:.1%}",
        'profit_factor': round(pf, 2),
        'trade_win_rate': f"{trade_wr:.1%}",
        'daily_win_rate': f"{daily_wr:.1%}",
        'total_trades': len(pos_df),
        'avg_positions': round(avg_positions, 1),
        'max_positions': int(max_positions),
        'avg_win': f"{avg_win:.2%}",
        'avg_loss': f"{avg_loss:.2%}",
        'years': round(years, 1),
    }

def regime_analysis(daily_df, spy_df):
    """Stratified performance by regime."""
    regime_map = classify_regime(spy_df)
    daily_df = daily_df.copy()
    daily_df['month'] = pd.to_datetime(daily_df['date']).dt.to_period('M')
    daily_df['regime'] = daily_df['month'].map(regime_map)

    results = {}
    for regime in ['green', 'red']:
        rdf = daily_df[daily_df['regime'] == regime]
        if len(rdf) < 20:
            continue
        rets = rdf['return']
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        results[regime] = {
            'sharpe': round(sharpe, 3),
            'mean_daily': f"{rets.mean():.4%}",
            'days': len(rdf),
        }

    if 'green' in results and 'red' in results:
        sg = results['green']['sharpe']
        sr = results['red']['sharpe']
        gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
        results['regime_gap'] = round(gap, 3)
        results['regime_gap_pass'] = gap <= 0.50

    return results

def strategy_correlation(pos_df):
    """Compute return correlation between strategies."""
    if len(pos_df) == 0:
        return {}

    strat_daily = pos_df.groupby(['entry_date', 'strategy'])['fwd_ret'].mean().unstack(fill_value=0)
    if strat_daily.shape[1] < 2:
        return {}

    corr = strat_daily.corr()
    # Average pairwise correlation
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    avg_corr = corr.where(mask).stack().mean()

    return {
        'avg_pairwise_corr': round(avg_corr, 3),
        'correlation_matrix': {str(k): round(v, 3) for k, v in corr.stack().items()},
    }

def per_strategy_stats(pos_df):
    """Stats per strategy."""
    stats = []
    for strat, sdf in pos_df.groupby('strategy'):
        n = len(sdf)
        wr = (sdf['fwd_ret'] > 0).mean()
        avg_ret = sdf['fwd_ret'].mean()
        sharpe = sdf['fwd_ret'].mean() / sdf['fwd_ret'].std() if sdf['fwd_ret'].std() > 0 else 0
        stats.append({
            'strategy': strat,
            'trades': n,
            'win_rate': f"{wr:.1%}",
            'avg_return': f"{avg_ret:.2%}",
            'sharpe_per_trade': round(sharpe, 3),
        })
    return sorted(stats, key=lambda x: x['trades'], reverse=True)

def permutation_test(pos_df, n_perms=500):
    """Portfolio-level permutation test: shuffle signal directions."""
    real_mean = pos_df['fwd_ret'].mean()
    count_better = 0
    for _ in range(n_perms):
        shuffled = pos_df['fwd_ret'].values.copy()
        signs = np.random.choice([-1, 1], size=len(shuffled))
        perm_mean = (shuffled * signs).mean()
        if perm_mean >= real_mean:
            count_better += 1
    p_value = count_better / n_perms
    return p_value

def yearly_performance(daily_df):
    """Annual returns breakdown."""
    daily_df = daily_df.copy()
    daily_df['year'] = pd.to_datetime(daily_df['date']).dt.year
    yearly = []
    for year, ydf in daily_df.groupby('year'):
        total_ret = (1 + ydf['return']).prod() - 1
        sharpe = ydf['return'].mean() / ydf['return'].std() * np.sqrt(252) if ydf['return'].std() > 0 else 0
        yearly.append({
            'year': year,
            'return': f"{total_ret:.1%}",
            'sharpe': round(sharpe, 2),
            'trading_days': len(ydf),
        })
    return yearly


def main():
    print("=" * 70)
    print("MULTI-STRATEGY PORTFOLIO BACKTEST v1")
    print(f"Testing combined performance of all proven signals")
    print(f"Universe: S&P 500 | Period: 2013-2026")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    tickers = get_sp500_tickers()
    df = load_price_data(tickers)
    spy_df = load_spy_data()

    # Compute features
    print("\n[2] Computing features (RSI, vol, MFI, OBV)...")
    df = compute_features(df)
    print(f"  Features computed for {df['ticker'].nunique()} tickers, {len(df)} rows")

    # Generate signals
    print("\n[3] Detecting signals across all strategies...")
    all_signals = []

    strategies = [
        ('oversold_bounce', lambda: signal_oversold_bounce(df, rsi_thresh=20, hold=10)),
        ('drop_3pct', lambda: signal_drop_3pct(df, drop_thresh=0.03, hold=10)),
        ('confluence_drop_vc', lambda: signal_confluence_drop_volcomp(df, hold=10)),
        ('vol_climax', lambda: signal_vol_climax(df, hold=5)),
        ('drop3_mfi_oversold', lambda: signal_drop3_mfi_oversold(df, hold=21)),
        ('volcomp_mfi', lambda: signal_volcomp_mfi_oversold(df, hold=10)),
        ('oversold_high_relvol', lambda: signal_oversold_high_relvol(df, hold=21)),
    ]

    for name, detector in strategies:
        sigs = detector()
        print(f"  {name}: {len(sigs)} signals")
        all_signals.append(sigs)

    all_signals_df = pd.concat(all_signals, ignore_index=True)
    print(f"\n  Total signals: {len(all_signals_df)} across {all_signals_df['strategy'].nunique()} strategies")

    # Simulate portfolio
    print("\n[4] Simulating portfolio (max 20 positions, max 10% per stock)...")
    daily_df, pos_df = simulate_portfolio(all_signals_df, max_positions=20)

    if len(daily_df) == 0:
        print("ERROR: No portfolio data generated")
        return

    # Compute metrics
    print("\n[5] Computing performance metrics...")
    metrics = compute_metrics(daily_df, pos_df)
    print(f"\n  PORTFOLIO METRICS:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")

    # Regime analysis
    print("\n[6] Regime analysis...")
    regime = regime_analysis(daily_df, spy_df)
    print(f"  Green regime Sharpe: {regime.get('green', {}).get('sharpe', 'N/A')}")
    print(f"  Red regime Sharpe: {regime.get('red', {}).get('sharpe', 'N/A')}")
    print(f"  Regime gap: {regime.get('regime_gap', 'N/A')} (pass: {regime.get('regime_gap_pass', 'N/A')})")

    # Strategy correlation
    print("\n[7] Strategy diversification...")
    corr = strategy_correlation(pos_df)
    print(f"  Avg pairwise correlation: {corr.get('avg_pairwise_corr', 'N/A')}")

    # Per-strategy breakdown
    print("\n[8] Per-strategy performance:")
    strat_stats = per_strategy_stats(pos_df)
    for s in strat_stats:
        print(f"    {s['strategy']}: {s['trades']} trades, WR {s['win_rate']}, avg ret {s['avg_return']}")

    # Permutation test
    print("\n[9] Permutation test (500 iterations)...")
    p_val = permutation_test(pos_df)
    print(f"  Permutation p-value: {p_val:.4f} ({'PASS' if p_val < 0.05 else 'FAIL'})")

    # Yearly performance
    print("\n[10] Yearly performance:")
    yearly = yearly_performance(daily_df)
    for y in yearly:
        print(f"    {y['year']}: {y['return']} (Sharpe {y['sharpe']})")

    # Save results
    report = {
        'metrics': metrics,
        'regime': regime,
        'strategy_stats': strat_stats,
        'permutation_p': p_val,
        'yearly': yearly,
        'diversification': {'avg_pairwise_corr': corr.get('avg_pairwise_corr', None)},
    }

    with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Save daily returns for further analysis
    daily_df.to_parquet(os.path.join(OUTPUT_DIR, 'daily_returns.parquet'))
    pos_df.to_parquet(os.path.join(OUTPUT_DIR, 'positions.parquet'))

    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Sharpe: {metrics['sharpe']}")
    print(f"  Sortino: {metrics['sortino']}")
    print(f"  CAGR: {metrics['cagr']}")
    print(f"  Max DD: {metrics['max_drawdown']}")
    print(f"  Win Rate: {metrics['trade_win_rate']}")
    print(f"  Profit Factor: {metrics['profit_factor']}")
    print(f"  Regime Gap: {regime.get('regime_gap', 'N/A')}")
    print(f"  Permutation p: {p_val:.4f}")
    print(f"  Total Trades: {metrics['total_trades']}")
    print(f"\n  Report saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()
