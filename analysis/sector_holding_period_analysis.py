#!/usr/bin/env python3
"""
Sector ETF Optimal Holding Period Analysis
==========================================
Analyzes optimal holding periods by sector and entry condition for options trading.
- 11 sector ETFs, 2020-2026 daily data
- RSI(14) < 35 bullish entries, RSI(14) > 70 bearish entries
- Forward returns at 1,2,3,5,7,10,15,20 day horizons
- Sharpe ratios, TP hit rates, regime stratification
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import sys

SECTORS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLU': 'Utilities',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLV': 'Healthcare',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLC': 'Communication Services',
    'XLRE': 'Real Estate'
}

HORIZONS = [1, 2, 3, 5, 7, 10, 15, 20]
TP_LEVELS = [5, 10, 15, 20, 30]  # percentage thresholds

RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 70
RSI_PERIOD = 14

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_sma(series, period=20):
    return series.rolling(window=period).mean()

def download_data():
    """Download all sector ETFs + SPY for regime classification."""
    tickers = list(SECTORS.keys()) + ['SPY']
    print(f"Downloading data for {len(tickers)} tickers...")

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start='2020-01-01', end='2026-08-21', progress=False)
            if len(df) > 100:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: INSUFFICIENT DATA ({len(df)} days)")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    return data

def analyze_sector(ticker, df, spy_df, direction='long'):
    """
    Analyze a single sector ETF.
    direction='long': RSI < 35 entries, long/call direction
    direction='short': RSI > 70 entries, short/put direction
    """
    close = df['Close'].copy()
    rsi = compute_rsi(close, RSI_PERIOD)

    # SPY regime: bull = close > 20-SMA
    spy_close = spy_df['Close'].reindex(close.index, method='ffill')
    spy_sma20 = compute_sma(spy_close, 20)
    is_bull = spy_close > spy_sma20

    # Entry signals
    if direction == 'long':
        entries = rsi < RSI_OVERSOLD
    else:
        entries = rsi > RSI_OVERBOUGHT

    entry_dates = close.index[entries]

    if len(entry_dates) < 3:
        return None

    results = {
        'ticker': ticker,
        'sector': SECTORS.get(ticker, ticker),
        'direction': direction,
        'n_entries': len(entry_dates),
        'horizons': {},
        'regime_analysis': {'bull': {}, 'bear': {}},
    }

    for h in HORIZONS:
        fwd_returns = []
        max_fwd_returns = []  # MFE within horizon
        regime_returns = {'bull': [], 'bear': []}
        tp_hits = {tp: 0 for tp in TP_LEVELS}

        for date in entry_dates:
            loc = close.index.get_loc(date)
            end_loc = min(loc + h, len(close) - 1)

            if end_loc <= loc:
                continue

            entry_price = close.iloc[loc]

            # Forward return at exactly horizon h
            if loc + h < len(close):
                exit_price = close.iloc[loc + h]
                if direction == 'long':
                    ret = (exit_price - entry_price) / entry_price * 100
                else:
                    ret = (entry_price - exit_price) / entry_price * 100
                fwd_returns.append(ret)

                # MFE within horizon (max favorable excursion)
                window = close.iloc[loc:loc+h+1]
                if direction == 'long':
                    mfe = (window.max() - entry_price) / entry_price * 100
                else:
                    mfe = (entry_price - window.min()) / entry_price * 100
                max_fwd_returns.append(mfe)

                # TP hit rates (did price ever reach TP% within horizon?)
                for tp in TP_LEVELS:
                    if mfe >= tp:
                        tp_hits[tp] += 1

                # Regime at entry
                if date in is_bull.index:
                    regime = 'bull' if is_bull.loc[date] else 'bear'
                    regime_returns[regime].append(ret)

        if len(fwd_returns) < 3:
            continue

        fwd_arr = np.array(fwd_returns)
        mfe_arr = np.array(max_fwd_returns)
        n = len(fwd_arr)

        # Annualized Sharpe (using 252 trading days)
        mean_ret = fwd_arr.mean()
        std_ret = fwd_arr.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(252 / h) if std_ret > 0 else 0

        # Sortino
        downside = fwd_arr[fwd_arr < 0]
        downside_std = downside.std() if len(downside) > 1 else std_ret
        sortino = (mean_ret / downside_std) * np.sqrt(252 / h) if downside_std > 0 else 0

        results['horizons'][h] = {
            'n_trades': n,
            'mean_return_pct': round(mean_ret, 3),
            'median_return_pct': round(np.median(fwd_arr), 3),
            'std_pct': round(std_ret, 3),
            'win_rate': round((fwd_arr > 0).mean() * 100, 1),
            'sharpe': round(sharpe, 3),
            'sortino': round(sortino, 3),
            'avg_mfe_pct': round(mfe_arr.mean(), 3),
            'median_mfe_pct': round(np.median(mfe_arr), 3),
            'p90_mfe_pct': round(np.percentile(mfe_arr, 90), 3),
            'tp_hit_rates': {tp: round(tp_hits[tp] / n * 100, 1) for tp in TP_LEVELS},
        }

        # Regime analysis
        for regime in ['bull', 'bear']:
            rr = np.array(regime_returns[regime])
            if len(rr) >= 3:
                r_mean = rr.mean()
                r_std = rr.std()
                r_sharpe = (r_mean / r_std) * np.sqrt(252 / h) if r_std > 0 else 0
                results['regime_analysis'][regime][h] = {
                    'n': len(rr),
                    'mean_return_pct': round(r_mean, 3),
                    'win_rate': round((rr > 0).mean() * 100, 1),
                    'sharpe': round(r_sharpe, 3),
                }

    # Find optimal horizon (max Sharpe)
    if results['horizons']:
        best_h = max(results['horizons'].keys(),
                     key=lambda h: results['horizons'][h]['sharpe'])
        results['optimal_horizon'] = best_h
        results['optimal_sharpe'] = results['horizons'][best_h]['sharpe']

        # Find optimal TP (highest TP with >50% hit rate at optimal horizon)
        tp_rates = results['horizons'][best_h]['tp_hit_rates']
        viable_tps = [tp for tp, rate in tp_rates.items() if rate >= 30]
        results['max_viable_tp'] = max(viable_tps) if viable_tps else 5

        # Find TP with best risk-adjusted expectation
        # Simple heuristic: TP% * hit_rate gives expected value
        best_ev_tp = max(TP_LEVELS, key=lambda tp: tp * tp_rates.get(tp, 0) / 100)
        results['best_ev_tp'] = best_ev_tp

    return results

def print_summary(all_results):
    """Print comprehensive summary tables."""

    print("\n" + "="*120)
    print("SECTOR ETF OPTIMAL HOLDING PERIOD ANALYSIS")
    print("="*120)

    # === LONG/CALL ENTRIES (RSI < 35) ===
    print(f"\n{'='*120}")
    print(f"LONG/CALL ENTRIES (RSI(14) < {RSI_OVERSOLD})")
    print(f"{'='*120}")

    long_results = [r for r in all_results if r and r['direction'] == 'long']

    if long_results:
        # Table 1: Optimal holding period per sector
        print(f"\n{'SECTOR OPTIMAL HOLDING PERIODS':^120}")
        print("-"*120)
        header = f"{'Ticker':<6} {'Sector':<25} {'#Entries':>8} {'Opt.Days':>9} {'Sharpe':>8} {'Sortino':>9} {'Mean%':>8} {'WR%':>6} {'MFE%':>8} {'BestTP':>7}"
        print(header)
        print("-"*120)

        for r in sorted(long_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            if 'optimal_horizon' not in r:
                continue
            oh = r['optimal_horizon']
            h_data = r['horizons'][oh]
            print(f"{r['ticker']:<6} {r['sector']:<25} {r['n_entries']:>8} {oh:>9} "
                  f"{h_data['sharpe']:>8.2f} {h_data['sortino']:>9.2f} "
                  f"{h_data['mean_return_pct']:>8.2f} {h_data['win_rate']:>6.1f} "
                  f"{h_data['avg_mfe_pct']:>8.2f} {r.get('best_ev_tp', 'N/A'):>7}")

        # Table 2: Sharpe by horizon for each sector
        print(f"\n{'SHARPE RATIO BY HOLDING PERIOD (LONG)':^120}")
        print("-"*120)
        header = f"{'Ticker':<6} " + "".join(f"{h:>8}d" for h in HORIZONS)
        print(header)
        print("-"*120)

        for r in sorted(long_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            row = f"{r['ticker']:<6} "
            for h in HORIZONS:
                if h in r['horizons']:
                    s = r['horizons'][h]['sharpe']
                    marker = " *" if h == r.get('optimal_horizon') else "  "
                    row += f"{s:>7.2f}{marker}"
                else:
                    row += f"{'N/A':>9}"
            print(row)
        print("  (* = optimal horizon)")

        # Table 3: TP Hit Rates at optimal horizon
        print(f"\n{'TP HIT RATES AT OPTIMAL HORIZON (LONG)':^120}")
        print("-"*120)
        header = f"{'Ticker':<6} {'OptDays':>7} " + "".join(f"  +{tp}%TP" for tp in TP_LEVELS)
        print(header)
        print("-"*120)

        for r in sorted(long_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            if 'optimal_horizon' not in r:
                continue
            oh = r['optimal_horizon']
            h_data = r['horizons'][oh]
            row = f"{r['ticker']:<6} {oh:>7} "
            for tp in TP_LEVELS:
                rate = h_data['tp_hit_rates'].get(tp, 0)
                row += f"{rate:>7.1f}%"
            print(row)

        # Table 4: Mean return by horizon
        print(f"\n{'MEAN RETURN % BY HOLDING PERIOD (LONG)':^120}")
        print("-"*120)
        header = f"{'Ticker':<6} " + "".join(f"{h:>8}d" for h in HORIZONS)
        print(header)
        print("-"*120)

        for r in sorted(long_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            row = f"{r['ticker']:<6} "
            for h in HORIZONS:
                if h in r['horizons']:
                    m = r['horizons'][h]['mean_return_pct']
                    row += f"{m:>8.2f}%"
                else:
                    row += f"{'N/A':>9}"
            print(row)

        # Table 5: Regime analysis
        print(f"\n{'REGIME ANALYSIS: BULL vs BEAR SHARPE AT OPTIMAL HORIZON (LONG)':^120}")
        print("-"*120)
        header = f"{'Ticker':<6} {'OptDays':>7} {'All Sharpe':>11} {'Bull Sharpe':>12} {'Bull WR%':>9} {'Bear Sharpe':>12} {'Bear WR%':>9} {'Regime Gap':>11}"
        print(header)
        print("-"*120)

        for r in sorted(long_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            if 'optimal_horizon' not in r:
                continue
            oh = r['optimal_horizon']
            h_data = r['horizons'][oh]
            bull = r['regime_analysis']['bull'].get(oh, {})
            bear = r['regime_analysis']['bear'].get(oh, {})

            bull_s = bull.get('sharpe', float('nan'))
            bear_s = bear.get('sharpe', float('nan'))
            bull_wr = bull.get('win_rate', float('nan'))
            bear_wr = bear.get('win_rate', float('nan'))

            if not np.isnan(bull_s) and not np.isnan(bear_s):
                gap = abs(bull_s - bear_s) / max(abs(bull_s), abs(bear_s), 0.01)
            else:
                gap = float('nan')

            print(f"{r['ticker']:<6} {oh:>7} {h_data['sharpe']:>11.2f} "
                  f"{bull_s:>12.2f} {bull_wr:>9.1f} "
                  f"{bear_s:>12.2f} {bear_wr:>9.1f} "
                  f"{gap:>11.2f}")

    # === SHORT/PUT ENTRIES (RSI > 70) ===
    print(f"\n\n{'='*120}")
    print(f"SHORT/PUT ENTRIES (RSI(14) > {RSI_OVERBOUGHT})")
    print(f"{'='*120}")

    short_results = [r for r in all_results if r and r['direction'] == 'short']

    if short_results:
        print(f"\n{'SECTOR OPTIMAL HOLDING PERIODS (SHORT/PUT)':^120}")
        print("-"*120)
        header = f"{'Ticker':<6} {'Sector':<25} {'#Entries':>8} {'Opt.Days':>9} {'Sharpe':>8} {'Sortino':>9} {'Mean%':>8} {'WR%':>6} {'MFE%':>8} {'BestTP':>7}"
        print(header)
        print("-"*120)

        for r in sorted(short_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            if 'optimal_horizon' not in r:
                continue
            oh = r['optimal_horizon']
            h_data = r['horizons'][oh]
            print(f"{r['ticker']:<6} {r['sector']:<25} {r['n_entries']:>8} {oh:>9} "
                  f"{h_data['sharpe']:>8.2f} {h_data['sortino']:>9.2f} "
                  f"{h_data['mean_return_pct']:>8.2f} {h_data['win_rate']:>6.1f} "
                  f"{h_data['avg_mfe_pct']:>8.2f} {r.get('best_ev_tp', 'N/A'):>7}")

        # Sharpe by horizon (short)
        print(f"\n{'SHARPE RATIO BY HOLDING PERIOD (SHORT/PUT)':^120}")
        print("-"*120)
        header = f"{'Ticker':<6} " + "".join(f"{h:>8}d" for h in HORIZONS)
        print(header)
        print("-"*120)

        for r in sorted(short_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            row = f"{r['ticker']:<6} "
            for h in HORIZONS:
                if h in r['horizons']:
                    s = r['horizons'][h]['sharpe']
                    marker = " *" if h == r.get('optimal_horizon') else "  "
                    row += f"{s:>7.2f}{marker}"
                else:
                    row += f"{'N/A':>9}"
            print(row)
        print("  (* = optimal horizon)")

        # TP hit rates (short)
        print(f"\n{'TP HIT RATES AT OPTIMAL HORIZON (SHORT/PUT)':^120}")
        print("-"*120)
        header = f"{'Ticker':<6} {'OptDays':>7} " + "".join(f"  +{tp}%TP" for tp in TP_LEVELS)
        print(header)
        print("-"*120)

        for r in sorted(short_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            if 'optimal_horizon' not in r:
                continue
            oh = r['optimal_horizon']
            h_data = r['horizons'][oh]
            row = f"{r['ticker']:<6} {oh:>7} "
            for tp in TP_LEVELS:
                rate = h_data['tp_hit_rates'].get(tp, 0)
                row += f"{rate:>7.1f}%"
            print(row)

    # === COMPARISON WITH CURRENT STRATEGY ===
    print(f"\n\n{'='*120}")
    print("COMPARISON: CURRENT 5-DAY / +30% TP vs OPTIMAL")
    print("="*120)

    print(f"\n{'Ticker':<6} {'Direction':<7} {'Curr 5d Sharpe':>15} {'Opt Horizon':>12} {'Opt Sharpe':>11} {'Sharpe Diff':>12} {'Curr 30%TP Hit':>15} {'Recommendation':<40}")
    print("-"*140)

    for r in all_results:
        if r is None or 'optimal_horizon' not in r:
            continue

        curr_sharpe = r['horizons'].get(5, {}).get('sharpe', float('nan'))
        opt_sharpe = r['optimal_sharpe']
        opt_h = r['optimal_horizon']

        curr_tp30 = r['horizons'].get(5, {}).get('tp_hit_rates', {}).get(30, 0)

        diff = opt_sharpe - curr_sharpe if not np.isnan(curr_sharpe) else float('nan')

        # Recommendation
        if opt_h < 5 and diff > 0.1:
            rec = f"SHORTEN to {opt_h}d, faster mean-reversion"
        elif opt_h > 5 and diff > 0.1:
            rec = f"EXTEND to {opt_h}d, more room to run"
        elif abs(diff) <= 0.1:
            rec = "5d is near-optimal, keep current"
        else:
            rec = f"Consider {opt_h}d (marginal improvement)"

        # TP recommendation
        best_tp = r.get('best_ev_tp', 30)
        if best_tp != 30:
            rec += f", TP->{best_tp}%"

        dir_label = 'LONG' if r['direction'] == 'long' else 'SHORT'
        print(f"{r['ticker']:<6} {dir_label:<7} {curr_sharpe:>15.2f} {opt_h:>12} {opt_sharpe:>11.2f} {diff:>12.2f} {curr_tp30:>14.1f}% {rec:<40}")

    # === MFE ANALYSIS ===
    print(f"\n\n{'='*120}")
    print("MFE (MAX FAVORABLE EXCURSION) ANALYSIS — HOW FAR DOES PRICE MOVE IN OUR FAVOR?")
    print("="*120)

    for direction in ['long', 'short']:
        dir_label = 'LONG/CALL' if direction == 'long' else 'SHORT/PUT'
        dir_results = [r for r in all_results if r and r['direction'] == direction]

        if not dir_results:
            continue

        print(f"\n{dir_label} ENTRIES — Average MFE% within horizon:")
        print("-"*100)
        header = f"{'Ticker':<6} " + "".join(f"{h:>8}d" for h in HORIZONS)
        print(header)
        print("-"*100)

        for r in sorted(dir_results, key=lambda x: x.get('optimal_sharpe', 0), reverse=True):
            row = f"{r['ticker']:<6} "
            for h in HORIZONS:
                if h in r['horizons']:
                    mfe = r['horizons'][h]['avg_mfe_pct']
                    row += f"{mfe:>7.2f}%"
                else:
                    row += f"{'N/A':>8}"
            print(row)


def main():
    print("="*80)
    print("SECTOR ETF OPTIMAL HOLDING PERIOD ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Sectors: {len(SECTORS)}")
    print(f"Entry conditions: RSI(14) < {RSI_OVERSOLD} (long), RSI(14) > {RSI_OVERBOUGHT} (short)")
    print(f"Horizons: {HORIZONS} days")
    print("="*80)

    # Download data
    data = download_data()

    if 'SPY' not in data:
        print("ERROR: Could not download SPY data for regime classification")
        sys.exit(1)

    spy_df = data['SPY']

    # Analyze each sector
    all_results = []

    for ticker in SECTORS:
        if ticker not in data:
            print(f"Skipping {ticker} — no data")
            continue

        print(f"\nAnalyzing {ticker} ({SECTORS[ticker]})...")

        # Long entries (RSI < 35)
        long_result = analyze_sector(ticker, data[ticker], spy_df, direction='long')
        if long_result:
            all_results.append(long_result)
            print(f"  Long entries: {long_result['n_entries']}, optimal horizon: {long_result.get('optimal_horizon', 'N/A')}d")
        else:
            print(f"  Long entries: insufficient RSI<{RSI_OVERSOLD} signals")

        # Short entries (RSI > 70)
        short_result = analyze_sector(ticker, data[ticker], spy_df, direction='short')
        if short_result:
            all_results.append(short_result)
            print(f"  Short entries: {short_result['n_entries']}, optimal horizon: {short_result.get('optimal_horizon', 'N/A')}d")
        else:
            print(f"  Short entries: insufficient RSI>{RSI_OVERBOUGHT} signals")

    # Print comprehensive summary
    print_summary(all_results)

    # Save raw results as JSON
    output_path = '/home/jupiter/Lvl3Quant/analysis/sector_holding_results.json'
    serializable = []
    for r in all_results:
        if r:
            sr = r.copy()
            sr['horizons'] = {str(k): v for k, v in sr['horizons'].items()}
            sr['regime_analysis'] = {
                regime: {str(k): v for k, v in hdata.items()}
                for regime, hdata in sr['regime_analysis'].items()
            }
            serializable.append(sr)

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {output_path}")


if __name__ == '__main__':
    main()
