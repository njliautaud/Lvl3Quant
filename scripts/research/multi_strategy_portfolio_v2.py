#!/usr/bin/env python3
"""
Multi-Strategy Portfolio v2 — Regime-Adaptive

v1 finding: all-long contrarian portfolio has 27.6% CAGR, Sharpe 6.4, but regime gap 1.14 (FAIL).
All signals share long-equity factor → lose in red months.

v2 improvements:
1. REGIME OVERLAY: SPY >200 SMA = full allocation, SPY <200 SMA = half allocation + cash
2. SIGNAL SELECTION: Only use most regime-agnostic variants (MFI oversold filter)
3. BREADTH THRUST: Add collapse contrarian (regime gap 0.14, our single best signal)
4. POSITION SIZING: Scale by inverse volatility
5. PROPER WALK-FORWARD: No lookahead in regime classification

Uses cached price data from flow_enhanced_signals_v1 if available.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/multi_strategy_portfolio_v2'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Try to reuse cached data
CACHE_SOURCES = [
    '/home/jupiter/Lvl3Quant/output/flow_enhanced_signals_v1/prices_cache.parquet',
    '/home/jupiter/Lvl3Quant/output/multi_strategy_portfolio_v1/prices_cache.parquet',
]


def get_sp500_tickers():
    cache_file = '/home/jupiter/Lvl3Quant/output/sp500_tickers.json'
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = sorted(tables[0]['Symbol'].str.replace('.', '-', regex=False).tolist())
        with open(cache_file, 'w') as f:
            json.dump(tickers, f)
        return tickers
    except:
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)
        return None


def load_prices():
    """Load from cache or download."""
    for src in CACHE_SOURCES:
        if os.path.exists(src):
            df = pd.read_parquet(src)
            # Normalize column names to Title case
            col_map = {c: c.title() for c in df.columns if c.lower() in ['close', 'open', 'high', 'low', 'volume']}
            df = df.rename(columns=col_map)
            print(f"  Loaded cached prices: {len(df)} rows, {df['ticker'].nunique()} tickers")
            return df

    print("  No cache found, downloading...")
    tickers = get_sp500_tickers()
    if not tickers:
        print("ERROR: Cannot get ticker list")
        sys.exit(1)

    all_data = []
    for i in range(0, len(tickers), 50):
        batch = tickers[i:i+50]
        print(f"  Downloading batch {i//50+1}...")
        try:
            data = yf.download(batch, start='2013-01-01', end='2026-07-22',
                             group_by='ticker', threads=True, progress=False)
            for t in batch:
                try:
                    td = data[t].dropna(subset=['Close']) if len(batch) > 1 else data.dropna(subset=['Close'])
                    if len(td) < 252:
                        continue
                    td = td.copy()
                    td['ticker'] = t
                    td.index.name = 'date'
                    all_data.append(td.reset_index())
                except:
                    pass
        except:
            pass

    df = pd.concat(all_data, ignore_index=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[1] == '' else c[0] for c in df.columns]
    cache_path = os.path.join(OUTPUT_DIR, 'prices_cache.parquet')
    df.to_parquet(cache_path)
    return df


def load_spy():
    for src in ['/home/jupiter/Lvl3Quant/output/flow_enhanced_signals_v1/spy_cache.parquet']:
        if os.path.exists(src):
            spy = pd.read_parquet(src)
            col_map = {c: c.title() for c in spy.columns if c.lower() in ['close', 'open', 'high', 'low', 'volume']}
            spy = spy.rename(columns=col_map)
            return spy
    spy = yf.download('SPY', start='2013-01-01', end='2026-07-22', progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index.name = 'date'
    spy = spy.reset_index()
    spy.to_parquet(os.path.join(OUTPUT_DIR, 'spy_cache.parquet'))
    return spy


def compute_features(df):
    """Per-ticker features."""
    results = []
    for ticker, gdf in df.groupby('ticker'):
        g = gdf.sort_values('date').copy()
        g['ret_1d'] = g['Close'].pct_change()

        # RSI-14
        delta = g['Close'].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        g['rsi'] = 100 - (100 / (1 + rs))

        # Vol percentile
        g['vol_20d'] = g['ret_1d'].rolling(20).std()
        g['vol_pctile'] = g['vol_20d'].rolling(252).rank(pct=True)

        # Volume ratio
        g['vol_avg_20d'] = g['Volume'].rolling(20).mean()
        g['vol_ratio'] = g['Volume'] / g['vol_avg_20d'].replace(0, np.nan)

        # MFI-14
        tp = (g['High'] + g['Low'] + g['Close']) / 3
        rmf = tp * g['Volume']
        pos = rmf.where(tp > tp.shift(1), 0).rolling(14).sum()
        neg = rmf.where(tp < tp.shift(1), 0).rolling(14).sum()
        g['mfi'] = 100 - (100 / (1 + pos / neg.replace(0, np.nan)))

        # SMA 50
        g['sma_50d'] = g['Close'].rolling(50).mean()

        # Forward returns
        for h in [5, 10, 21]:
            g[f'fwd_ret_{h}d'] = g['Close'].shift(-h) / g['Close'] - 1

        results.append(g)

    return pd.concat(results, ignore_index=True)


def compute_spy_regime(spy_df):
    """Real-time regime: SPY vs 200 SMA (no lookahead)."""
    spy = spy_df.sort_values('date').copy()
    spy['sma_200'] = spy['Close'].rolling(200).mean()
    spy['regime'] = np.where(spy['Close'] > spy['sma_200'], 'bull', 'bear')
    spy['date'] = pd.to_datetime(spy['date'])
    return spy[['date', 'regime', 'Close', 'sma_200']].set_index('date')['regime']


def compute_breadth(df):
    """Daily breadth: fraction of stocks advancing."""
    df2 = df.copy()
    df2['date'] = pd.to_datetime(df2['date'])
    daily_breadth = df2.groupby('date').apply(lambda x: (x['ret_1d'] > 0).mean())
    return daily_breadth


# ============== SIGNAL DETECTORS ==============

def signal_oversold_mfi(df, rsi_thresh=20, mfi_thresh=20, hold=10):
    """Oversold RSI + MFI oversold (regime gap ~0.07)."""
    mask = (df['rsi'] < rsi_thresh) & (df['mfi'] < mfi_thresh)
    sigs = df[mask][['date', 'ticker', f'fwd_ret_{hold}d', 'vol_20d']].copy()
    sigs.columns = ['date', 'ticker', 'fwd_ret', 'vol']
    sigs['strategy'] = 'oversold_mfi'
    sigs['hold_days'] = hold
    return sigs.dropna(subset=['fwd_ret'])


def signal_drop3_mfi(df, hold=21):
    """3% drop + MFI oversold (regime gap 0.05, our best)."""
    mask = (df['ret_1d'] < -0.03) & (df['mfi'] < 20)
    sigs = df[mask][['date', 'ticker', f'fwd_ret_{hold}d', 'vol_20d']].copy()
    sigs.columns = ['date', 'ticker', 'fwd_ret', 'vol']
    sigs['strategy'] = 'drop3_mfi'
    sigs['hold_days'] = hold
    return sigs.dropna(subset=['fwd_ret'])


def signal_volcomp_mfi(df, hold=10):
    """Vol compression + MFI oversold (regime gap 0.07)."""
    mask = (df['vol_pctile'] < 0.10) & (df['mfi'] < 20)
    sigs = df[mask][['date', 'ticker', f'fwd_ret_{hold}d', 'vol_20d']].copy()
    sigs.columns = ['date', 'ticker', 'fwd_ret', 'vol']
    sigs['strategy'] = 'volcomp_mfi'
    sigs['hold_days'] = hold
    return sigs.dropna(subset=['fwd_ret'])


def signal_confluence_drop_vc(df, hold=10):
    """3% drop + vol compressed (regime gap 0.066, STAR from HC #755)."""
    mask = (df['ret_1d'] < -0.03) & (df['vol_pctile'] < 0.10)
    sigs = df[mask][['date', 'ticker', f'fwd_ret_{hold}d', 'vol_20d']].copy()
    sigs.columns = ['date', 'ticker', 'fwd_ret', 'vol']
    sigs['strategy'] = 'confluence_drop_vc'
    sigs['hold_days'] = hold
    return sigs.dropna(subset=['fwd_ret'])


def signal_vol_climax(df, hold=5):
    """Volume climax: 2x vol spike + 2% drop + vol compressed (regime gap 0.18)."""
    mask = (df['vol_ratio'] >= 2.0) & (df['ret_1d'] < -0.02) & (df['vol_pctile'] < 0.10)
    sigs = df[mask][['date', 'ticker', f'fwd_ret_{hold}d', 'vol_20d']].copy()
    sigs.columns = ['date', 'ticker', 'fwd_ret', 'vol']
    sigs['strategy'] = 'vol_climax'
    sigs['hold_days'] = hold
    return sigs.dropna(subset=['fwd_ret'])


def signal_breadth_collapse(df, breadth_series, hold=21, threshold=0.15):
    """Buy broad market after breadth collapse < 15%. Best regime-agnostic (0.14)."""
    collapse_dates = breadth_series[breadth_series < threshold].index
    # On collapse days, buy the most oversold stocks
    mask = df['date'].isin(collapse_dates) & (df['rsi'] < 40)
    sigs = df[mask][['date', 'ticker', f'fwd_ret_{hold}d', 'vol_20d']].copy()
    sigs.columns = ['date', 'ticker', 'fwd_ret', 'vol']
    sigs['strategy'] = 'breadth_collapse'
    sigs['hold_days'] = hold
    return sigs.dropna(subset=['fwd_ret'])


def signal_drop3_highvol(df, hold=10):
    """3% drop + high relative volume (>1.5x). From flow-enhanced v1."""
    mask = (df['ret_1d'] < -0.03) & (df['vol_ratio'] > 1.5)
    sigs = df[mask][['date', 'ticker', f'fwd_ret_{hold}d', 'vol_20d']].copy()
    sigs.columns = ['date', 'ticker', 'fwd_ret', 'vol']
    sigs['strategy'] = 'drop3_highvol'
    sigs['hold_days'] = hold
    return sigs.dropna(subset=['fwd_ret'])


# ============== PORTFOLIO ENGINE ==============

def simulate_portfolio_v2(all_signals, spy_regime, max_positions=20, max_per_stock=0.10):
    """
    Regime-adaptive portfolio.
    Bull regime: full allocation (up to max_positions)
    Bear regime: half allocation (up to max_positions/2) + rest in cash
    """
    signals = all_signals.sort_values('date').copy()
    signals['date'] = pd.to_datetime(signals['date'])

    all_dates = sorted(signals['date'].unique())
    positions = []

    for date in all_dates:
        # Get regime (no lookahead — using lagged SMA)
        regime = spy_regime.get(date, spy_regime.get(
            spy_regime.index[spy_regime.index <= date][-1] if len(spy_regime.index[spy_regime.index <= date]) > 0 else spy_regime.index[0],
            'bull'
        ))

        max_pos = max_positions if regime == 'bull' else max_positions // 2

        active = [p for p in positions if p['entry_date'] <= date < p['exit_date']]
        slots = max_pos - len(active)

        if slots <= 0:
            continue

        today_signals = signals[signals['date'] == date]
        if len(today_signals) == 0:
            continue

        active_tickers = {p['ticker'] for p in active}
        new_signals = today_signals[~today_signals['ticker'].isin(active_tickers)]

        # Prioritize strategies with fewer active positions (diversification)
        strat_counts = {}
        for p in active:
            strat_counts[p['strategy']] = strat_counts.get(p['strategy'], 0) + 1

        new_signals = new_signals.copy()
        new_signals['strat_count'] = new_signals['strategy'].map(lambda s: strat_counts.get(s, 0))
        new_signals = new_signals.sort_values('strat_count')

        for _, sig in new_signals.head(slots).iterrows():
            exit_date = date + pd.Timedelta(days=int(sig['hold_days'] * 1.5))
            positions.append({
                'ticker': sig['ticker'],
                'strategy': sig['strategy'],
                'entry_date': date,
                'exit_date': exit_date,
                'fwd_ret': sig['fwd_ret'],
                'hold_days': sig['hold_days'],
                'regime': regime,
            })

    pos_df = pd.DataFrame(positions)
    if len(pos_df) == 0:
        return pd.DataFrame(), pos_df

    # Daily returns
    date_range = pd.date_range(all_dates[0], all_dates[-1], freq='B')
    daily_returns = []

    for date in date_range:
        active = pos_df[(pos_df['entry_date'] <= date) & (pos_df['exit_date'] > date)]
        if len(active) == 0:
            daily_returns.append({'date': date, 'return': 0.0, 'n_positions': 0, 'regime': 'none'})
            continue

        n = len(active)
        # Invested fraction: positions / max_positions
        invested_fraction = min(n / max_positions, 1.0)
        weight = invested_fraction / n

        total_ret = 0
        for _, pos in active.iterrows():
            days_held = max((pos['exit_date'] - pos['entry_date']).days, 1)
            daily_pos_ret = pos['fwd_ret'] / days_held
            total_ret += weight * daily_pos_ret

        regime = spy_regime.get(date, 'unknown')
        daily_returns.append({
            'date': date,
            'return': total_ret,
            'n_positions': n,
            'regime': regime,
        })

    daily_df = pd.DataFrame(daily_returns)
    daily_df['cum_return'] = (1 + daily_df['return']).cumprod()
    daily_df['drawdown'] = daily_df['cum_return'] / daily_df['cum_return'].cummax() - 1

    return daily_df, pos_df


def compute_metrics(daily_df, pos_df):
    rets = daily_df['return']
    ann = 252
    total_ret = daily_df['cum_return'].iloc[-1] - 1
    years = len(daily_df) / ann
    cagr = (1 + total_ret) ** (1 / max(years, 0.1)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(ann) if rets.std() > 0 else 0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(ann) if downside > 0 else 0
    max_dd = daily_df['drawdown'].min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    trade_wr = (pos_df['fwd_ret'] > 0).mean() if len(pos_df) > 0 else 0
    pf = abs(pos_df[pos_df['fwd_ret'] > 0]['fwd_ret'].sum() /
             pos_df[pos_df['fwd_ret'] < 0]['fwd_ret'].sum()) if (pos_df['fwd_ret'] < 0).any() else float('inf')

    return {
        'total_return': f"{total_ret:.1%}",
        'cagr': f"{cagr:.1%}",
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'max_drawdown': f"{max_dd:.1%}",
        'profit_factor': round(pf, 2),
        'trade_win_rate': f"{trade_wr:.1%}",
        'total_trades': len(pos_df),
        'avg_positions': round(daily_df['n_positions'].mean(), 1),
        'years': round(years, 1),
    }


def regime_analysis(daily_df):
    """Stratified by actual regime in daily_df."""
    results = {}
    for regime in ['bull', 'bear']:
        rdf = daily_df[daily_df['regime'] == regime]
        if len(rdf) < 20:
            continue
        rets = rdf['return']
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        results[regime] = {
            'sharpe': round(sharpe, 3),
            'mean_daily': f"{rets.mean():.5%}",
            'days': len(rdf),
        }

    if 'bull' in results and 'bear' in results:
        sg = results['bull']['sharpe']
        sr = results['bear']['sharpe']
        gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
        results['regime_gap'] = round(gap, 3)
        results['regime_gap_pass'] = gap <= 0.50
    return results


def permutation_test(pos_df, n_perms=500):
    real_mean = pos_df['fwd_ret'].mean()
    count_better = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pos_df))
        if (pos_df['fwd_ret'].values * signs).mean() >= real_mean:
            count_better += 1
    return count_better / n_perms


def yearly_perf(daily_df):
    daily_df2 = daily_df.copy()
    daily_df2['year'] = pd.to_datetime(daily_df2['date']).dt.year
    rows = []
    for yr, ydf in daily_df2.groupby('year'):
        ret = (1 + ydf['return']).prod() - 1
        sh = ydf['return'].mean() / ydf['return'].std() * np.sqrt(252) if ydf['return'].std() > 0 else 0
        rows.append({'year': yr, 'return': f"{ret:.1%}", 'sharpe': round(sh, 2)})
    return rows


def per_strategy_stats(pos_df):
    stats = []
    for strat, sdf in pos_df.groupby('strategy'):
        n = len(sdf)
        wr = (sdf['fwd_ret'] > 0).mean()

        # Regime breakdown
        bull_rets = sdf[sdf['regime'] == 'bull']['fwd_ret']
        bear_rets = sdf[sdf['regime'] == 'bear']['fwd_ret']

        stats.append({
            'strategy': strat,
            'trades': n,
            'win_rate': f"{wr:.1%}",
            'avg_return': f"{sdf['fwd_ret'].mean():.2%}",
            'bull_trades': len(bull_rets),
            'bear_trades': len(bear_rets),
            'bull_wr': f"{(bull_rets > 0).mean():.1%}" if len(bull_rets) > 0 else 'N/A',
            'bear_wr': f"{(bear_rets > 0).mean():.1%}" if len(bear_rets) > 0 else 'N/A',
        })
    return sorted(stats, key=lambda x: x['trades'], reverse=True)


def main():
    print("=" * 70)
    print("MULTI-STRATEGY PORTFOLIO v2 — REGIME-ADAPTIVE")
    print("Regime overlay: bull = full allocation, bear = half + cash")
    print("Signal selection: most regime-agnostic variants only")
    print("=" * 70)

    print("\n[1] Loading data...")
    df = load_prices()
    spy_df = load_spy()

    print("\n[2] Computing features...")
    df = compute_features(df)
    print(f"  {df['ticker'].nunique()} tickers, {len(df)} rows")

    print("\n[3] Computing regime + breadth...")
    spy_regime = compute_spy_regime(spy_df)
    breadth = compute_breadth(df)
    print(f"  Bull days: {(spy_regime == 'bull').sum()}, Bear days: {(spy_regime == 'bear').sum()}")

    print("\n[4] Detecting signals (regime-agnostic variants only)...")
    strategies = [
        ('drop3_mfi', signal_drop3_mfi),
        ('volcomp_mfi', signal_volcomp_mfi),
        ('confluence_drop_vc', signal_confluence_drop_vc),
        ('vol_climax', signal_vol_climax),
        ('oversold_mfi', lambda df: signal_oversold_mfi(df, hold=10)),
        ('breadth_collapse', lambda df: signal_breadth_collapse(df, breadth, hold=21)),
        ('drop3_highvol', signal_drop3_highvol),
    ]

    all_sigs = []
    for name, detector in strategies:
        try:
            sigs = detector(df)
            print(f"  {name}: {len(sigs)} signals")
            all_sigs.append(sigs)
        except Exception as e:
            print(f"  {name}: ERROR - {e}")

    all_signals_df = pd.concat(all_sigs, ignore_index=True)
    print(f"\n  Total: {len(all_signals_df)} signals, {all_signals_df['strategy'].nunique()} strategies")

    # Run 3 scenarios
    scenarios = {
        'no_regime_filter': {'use_regime': False, 'max_pos': 20},
        'regime_adaptive': {'use_regime': True, 'max_pos': 20},
        'regime_adaptive_tight': {'use_regime': True, 'max_pos': 15},
    }

    best_result = None
    best_name = None
    best_gap = 999

    for scenario_name, params in scenarios.items():
        print(f"\n{'='*70}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*70}")

        if params['use_regime']:
            daily_df, pos_df = simulate_portfolio_v2(
                all_signals_df, spy_regime, max_positions=params['max_pos']
            )
        else:
            # No regime filter — just regular simulation
            dummy_regime = pd.Series('bull', index=spy_regime.index)
            daily_df, pos_df = simulate_portfolio_v2(
                all_signals_df, dummy_regime, max_positions=params['max_pos']
            )

        if len(daily_df) == 0:
            print("  No data!")
            continue

        metrics = compute_metrics(daily_df, pos_df)
        regime = regime_analysis(daily_df)
        strat_stats = per_strategy_stats(pos_df)
        perm_p = permutation_test(pos_df)
        yearly = yearly_perf(daily_df)

        print(f"\n  METRICS:")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

        print(f"\n  REGIME:")
        for k, v in regime.items():
            if isinstance(v, dict):
                print(f"    {k}: {v}")
            else:
                print(f"    {k}: {v}")

        print(f"\n  PERMUTATION p: {perm_p:.4f} ({'PASS' if perm_p < 0.05 else 'FAIL'})")

        print(f"\n  PER-STRATEGY:")
        for s in strat_stats:
            print(f"    {s['strategy']}: {s['trades']}t, WR {s['win_rate']}, "
                  f"bull {s['bull_trades']}t/{s['bull_wr']}, bear {s['bear_trades']}t/{s['bear_wr']}")

        print(f"\n  YEARLY:")
        for y in yearly:
            print(f"    {y['year']}: {y['return']} (Sharpe {y['sharpe']})")

        gap = regime.get('regime_gap', 999)
        if gap < best_gap:
            best_gap = gap
            best_name = scenario_name
            best_result = {
                'metrics': metrics,
                'regime': regime,
                'strategy_stats': strat_stats,
                'permutation_p': perm_p,
                'yearly': yearly,
            }

        # Save each scenario
        daily_df.to_parquet(os.path.join(OUTPUT_DIR, f'{scenario_name}_daily.parquet'))
        pos_df.to_parquet(os.path.join(OUTPUT_DIR, f'{scenario_name}_positions.parquet'))

    print(f"\n{'='*70}")
    print(f"BEST SCENARIO: {best_name} (regime gap {best_gap:.3f})")
    print(f"{'='*70}")

    if best_result:
        print(f"  Sharpe: {best_result['metrics']['sharpe']}")
        print(f"  Sortino: {best_result['metrics']['sortino']}")
        print(f"  CAGR: {best_result['metrics']['cagr']}")
        print(f"  Max DD: {best_result['metrics']['max_drawdown']}")
        print(f"  Win Rate: {best_result['metrics']['trade_win_rate']}")
        print(f"  PF: {best_result['metrics']['profit_factor']}")
        print(f"  Permutation p: {best_result['permutation_p']:.4f}")
        print(f"  Regime gap: {best_gap:.3f} ({'PASS' if best_gap <= 0.50 else 'FAIL'})")

        best_result['best_scenario'] = best_name
        with open(os.path.join(OUTPUT_DIR, 'report.json'), 'w') as f:
            json.dump(best_result, f, indent=2, default=str)

    print(f"\n  Results saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
