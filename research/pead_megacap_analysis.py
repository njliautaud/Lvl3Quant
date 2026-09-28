#!/usr/bin/env python3
"""
PEAD Analysis for This Week's Mega-Cap Earnings
================================================
Quantify historical post-earnings drift for V, MSFT, META, AAPL, AMZN.
- How much do they drift after beats?
- How long does the drift last?
- What's the optimal entry timing (day +1, +2)?
- What's the optimal hold period?
- Bull call spread sizing analysis.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json, os, warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

TICKERS = ['V', 'MSFT', 'META', 'AAPL', 'AMZN']
CAPITAL = 645.0
RESULTS_DIR = '/home/nick/Lvl3Quant/research/findings'
os.makedirs(RESULTS_DIR, exist_ok=True)


def get_earnings_history(ticker):
    """Get earnings history from yfinance."""
    stock = yf.Ticker(ticker)
    try:
        earnings = stock.earnings_dates
        if earnings is not None and len(earnings) > 0:
            return earnings
    except:
        pass
    return None


def analyze_pead(ticker, prices, earnings_dates):
    """Analyze post-earnings drift for a single ticker."""
    results = []

    for date in earnings_dates:
        # Find the trading day index
        if date.date() not in prices.index:
            # Find nearest trading day
            close_dates = prices.index[prices.index <= str(date.date())]
            if len(close_dates) == 0:
                continue
            report_day = close_dates[-1]
        else:
            report_day = date.date() if isinstance(date, pd.Timestamp) else date

        idx = prices.index.get_loc(report_day) if report_day in prices.index else None
        if idx is None:
            continue

        # Need at least 20 days after earnings
        if idx + 20 >= len(prices):
            continue

        # Pre-earnings price (day before report for PM reports, report day for AM)
        pre_price = prices.iloc[idx]

        # Post-earnings returns at various horizons
        drift = {}
        for d in [1, 2, 3, 5, 10, 15, 20]:
            if idx + d < len(prices):
                drift[f'ret_{d}d'] = (prices.iloc[idx + d] / pre_price - 1) * 100

        # Earnings gap (day 1 open vs prior close)
        if idx + 1 < len(prices):
            drift['gap_1d'] = drift.get('ret_1d', 0)  # Approximate as day 1 close

        # Determine beat/miss (we'll use day-1 return as proxy)
        day1_ret = drift.get('ret_1d', 0)
        drift['beat'] = day1_ret > 0.5  # Gap up > 0.5% = likely beat
        drift['miss'] = day1_ret < -0.5
        drift['flat'] = not drift['beat'] and not drift['miss']
        drift['report_date'] = str(report_day)
        drift['day1_ret'] = day1_ret

        results.append(drift)

    return results


def compute_pead_stats(results, ticker):
    """Compute PEAD statistics."""
    df = pd.DataFrame(results)
    if len(df) == 0:
        return None

    beats = df[df['beat'] == True]
    misses = df[df['miss'] == True]

    stats = {
        'ticker': ticker,
        'total_earnings': len(df),
        'beats': len(beats),
        'misses': len(misses),
        'beat_rate': round(len(beats) / len(df) * 100, 1) if len(df) > 0 else 0
    }

    # Beat drift analysis
    if len(beats) >= 3:
        for d in [1, 2, 3, 5, 10, 15, 20]:
            col = f'ret_{d}d'
            if col in beats.columns:
                vals = beats[col].dropna()
                if len(vals) > 0:
                    stats[f'beat_drift_{d}d_mean'] = round(vals.mean(), 2)
                    stats[f'beat_drift_{d}d_median'] = round(vals.median(), 2)
                    stats[f'beat_drift_{d}d_std'] = round(vals.std(), 2)
                    stats[f'beat_drift_{d}d_winrate'] = round((vals > 0).mean() * 100, 1)

        # Additional drift (day 2-5 return AFTER day 1 gap)
        if 'ret_1d' in beats.columns and 'ret_5d' in beats.columns:
            residual_drift = beats['ret_5d'] - beats['ret_1d']
            stats['beat_residual_drift_2_5d_mean'] = round(residual_drift.mean(), 2)
            stats['beat_residual_drift_2_5d_winrate'] = round((residual_drift > 0).mean() * 100, 1)

        if 'ret_1d' in beats.columns and 'ret_10d' in beats.columns:
            residual_drift = beats['ret_10d'] - beats['ret_1d']
            stats['beat_residual_drift_2_10d_mean'] = round(residual_drift.mean(), 2)
            stats['beat_residual_drift_2_10d_winrate'] = round((residual_drift > 0).mean() * 100, 1)

    # Miss drift analysis
    if len(misses) >= 3:
        for d in [1, 2, 3, 5, 10]:
            col = f'ret_{d}d'
            if col in misses.columns:
                vals = misses[col].dropna()
                if len(vals) > 0:
                    stats[f'miss_drift_{d}d_mean'] = round(vals.mean(), 2)

    # Recent earnings (last 4 quarters)
    recent = df.sort_values('report_date', ascending=False).head(4)
    stats['recent_earnings'] = []
    for _, row in recent.iterrows():
        stats['recent_earnings'].append({
            'date': row['report_date'],
            'day1_return': round(row.get('day1_ret', 0), 2),
            'day5_return': round(row.get('ret_5d', 0), 2),
            'beat': bool(row.get('beat', False))
        })

    return stats


def optimal_entry_analysis(results, ticker):
    """Find optimal entry timing and holding period for PEAD trades."""
    df = pd.DataFrame(results)
    beats = df[df['beat'] == True]

    if len(beats) < 5:
        return None

    analysis = {'ticker': ticker}

    # Optimal entry: compare buying at day+1 open vs day+2
    for entry_day in [1, 2]:
        for hold_days in [3, 5, 7, 10]:
            exit_day = entry_day + hold_days
            entry_col = f'ret_{entry_day}d'
            exit_col = f'ret_{exit_day}d' if f'ret_{exit_day}d' in beats.columns else None

            if entry_col in beats.columns and exit_col and exit_col in beats.columns:
                trade_ret = beats[exit_col] - beats[entry_col]
                analysis[f'entry_d{entry_day}_hold_{hold_days}d'] = {
                    'mean_ret': round(trade_ret.mean(), 2),
                    'median_ret': round(trade_ret.median(), 2),
                    'winrate': round((trade_ret > 0).mean() * 100, 1),
                    'n_trades': len(trade_ret.dropna())
                }

    return analysis


def spread_sizing(ticker, price, stats):
    """Estimate optimal bull call spread parameters."""
    if not stats:
        return None

    # Use median 5-day beat drift to size the spread
    drift_5d = stats.get('beat_drift_5d_median', 0)
    drift_10d = stats.get('beat_drift_10d_median', 0)

    expected_move_5d = price * drift_5d / 100
    expected_move_10d = price * drift_10d / 100

    spread_widths = [2.5, 5.0, 7.5, 10.0]
    sizing = {'ticker': ticker, 'current_price': round(price, 2)}

    for width in spread_widths:
        # ATM spread: buy at current price, sell at current + width
        long_strike = round(price / 2.5) * 2.5  # Round to nearest $2.50
        short_strike = long_strike + width

        # Rough cost estimate: ~40% of width for ATM 10-14 DTE spread
        est_cost = width * 100 * 0.40
        max_profit = width * 100 - est_cost

        # Probability of max profit (stock > short strike) based on historical drift
        need_move = short_strike - price
        need_move_pct = need_move / price * 100

        # Use historical beat drift to estimate probability
        beat_5d_mean = stats.get('beat_drift_5d_mean', 0)
        beat_5d_std = stats.get('beat_drift_5d_std', 10)

        if beat_5d_std > 0:
            from scipy.stats import norm
            prob_itm = 1 - norm.cdf(need_move_pct, loc=beat_5d_mean, scale=beat_5d_std)
        else:
            prob_itm = 0.5

        ev = prob_itm * max_profit - (1 - prob_itm) * est_cost

        sizing[f'spread_{width}'] = {
            'long_strike': long_strike,
            'short_strike': short_strike,
            'est_cost': round(est_cost, 0),
            'max_profit': round(max_profit, 0),
            'need_move_pct': round(need_move_pct, 2),
            'prob_itm': round(prob_itm * 100, 1),
            'expected_value': round(ev, 0),
            'affordable': est_cost <= 300
        }

    return sizing


def main():
    print("=" * 70)
    print("PEAD MEGA-CAP ANALYSIS — This Week's Earnings")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    all_results = {}

    for ticker in TICKERS:
        print(f"\n{'='*50}")
        print(f"Analyzing {ticker}")
        print(f"{'='*50}")

        # Download price data
        prices_raw = yf.download(ticker, start='2020-01-01', end='2026-07-28',
                                  auto_adjust=True, progress=False)
        if isinstance(prices_raw.columns, pd.MultiIndex):
            prices = prices_raw['Close'].iloc[:, 0]
        else:
            prices = prices_raw['Close']
        prices = prices.dropna()

        # Get earnings dates
        earnings = get_earnings_history(ticker)
        if earnings is None or len(earnings) == 0:
            print(f"  No earnings data for {ticker}")
            continue

        # Filter to dates we have prices for
        earnings_dates = [d for d in earnings.index
                         if d.date() >= pd.Timestamp('2020-01-01').date()
                         and d.date() <= pd.Timestamp('2026-07-27').date()]
        print(f"  Found {len(earnings_dates)} earnings events since 2020")

        # Analyze PEAD
        pead_results = analyze_pead(ticker, prices, earnings_dates)
        print(f"  Analyzed {len(pead_results)} earnings events")

        # Compute statistics
        stats = compute_pead_stats(pead_results, ticker)
        if stats:
            print(f"  Beat rate: {stats['beat_rate']}%")
            print(f"  Beat drift (5d): mean={stats.get('beat_drift_5d_mean','N/A')}%, "
                  f"median={stats.get('beat_drift_5d_median','N/A')}%, "
                  f"WR={stats.get('beat_drift_5d_winrate','N/A')}%")
            print(f"  Beat drift (10d): mean={stats.get('beat_drift_10d_mean','N/A')}%, "
                  f"WR={stats.get('beat_drift_10d_winrate','N/A')}%")
            print(f"  Residual drift (d2-5): mean={stats.get('beat_residual_drift_2_5d_mean','N/A')}%, "
                  f"WR={stats.get('beat_residual_drift_2_5d_winrate','N/A')}%")
            print(f"  Residual drift (d2-10): mean={stats.get('beat_residual_drift_2_10d_mean','N/A')}%, "
                  f"WR={stats.get('beat_residual_drift_2_10d_winrate','N/A')}%")

            if stats.get('recent_earnings'):
                print(f"  Recent earnings:")
                for e in stats['recent_earnings']:
                    beat_str = "BEAT" if e['beat'] else "MISS"
                    print(f"    {e['date']}: {beat_str}, day1={e['day1_return']:+.1f}%, day5={e['day5_return']:+.1f}%")

        # Optimal entry analysis
        entry_analysis = optimal_entry_analysis(pead_results, ticker)
        if entry_analysis:
            print(f"\n  Optimal entry timing (post-beat):")
            for k, v in entry_analysis.items():
                if isinstance(v, dict):
                    print(f"    {k}: mean={v['mean_ret']:+.2f}%, WR={v['winrate']}%, n={v['n_trades']}")

        # Spread sizing
        current_price = float(prices.iloc[-1])
        sizing = spread_sizing(ticker, current_price, stats)
        if sizing:
            print(f"\n  Bull call spread sizing (price=${current_price:.2f}):")
            for k, v in sizing.items():
                if isinstance(v, dict):
                    affordable = "✅" if v['affordable'] else "❌"
                    print(f"    {k}: cost=${v['est_cost']}, max_profit=${v['max_profit']}, "
                          f"EV=${v['expected_value']}, prob={v['prob_itm']}% {affordable}")

        all_results[ticker] = {
            'pead_stats': stats,
            'entry_analysis': entry_analysis,
            'spread_sizing': sizing
        }

    # Save results
    out = os.path.join(RESULTS_DIR, 'pead_megacap_earnings_week.json')
    with open(out, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"RESULTS SAVED: {out}")

    # Summary ranking
    print(f"\n--- PEAD RANKING (best drift post-beat) ---")
    ranked = []
    for ticker, data in all_results.items():
        stats = data.get('pead_stats')
        if stats:
            drift_5d = stats.get('beat_drift_5d_mean', 0)
            wr_5d = stats.get('beat_drift_5d_winrate', 0)
            residual = stats.get('beat_residual_drift_2_5d_mean', 0)
            ranked.append((ticker, drift_5d, wr_5d, residual))

    ranked.sort(key=lambda x: x[3], reverse=True)  # Sort by residual drift (the tradeable part)
    for ticker, drift, wr, resid in ranked:
        print(f"  {ticker}: 5d total drift={drift:+.2f}%, WR={wr}%, "
              f"residual d2-5 drift={resid:+.2f}% (this is the PEAD alpha)")

    print(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
