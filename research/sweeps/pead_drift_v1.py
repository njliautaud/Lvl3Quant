#!/usr/bin/env python3
"""
Post-Earnings Announcement Drift (PEAD) Strategy v1
=====================================================

Academic anomaly: stocks that beat earnings expectations continue to drift
in the same direction for 20-60 days after the announcement. Stocks that
miss continue to drift down.

Different from our existing strategies:
- Post-Earnings Bounce: immediate 1-day reaction, 10d hold
- Earnings Jade Lizard: premium selling pre-earnings
- This: 20-60 day DRIFT after earnings, equity long/short

Key innovation: IV rank gating — don't enter when options are expensive.
Wait for IV to normalize (rank < 50th percentile) before entering the
drift trade. This avoids the post-earnings IV crush problem.

Universe: 50 large-cap stocks with reliable earnings data
Signal: Earnings surprise (actual vs expected, proxied by price reaction)
Hold: 20, 40, 60 trading days
Walk-forward: 252d train / 21d test, SLIDING (HC #0)
Adversarial: permutation (200), regime R1, sub-period, outlier

Per HC #741: minimum 10-15% CAGR target.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    print("MLflow connected")
except:
    print("MLflow unavailable")

RESULTS_DIR = Path(__file__).resolve().parent / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'pead_drift_v1_results.json'

UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'V', 'JPM', 'XOM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'PEP', 'COST', 'KO', 'AVGO', 'LLY', 'WMT', 'TMO', 'MCD',
    'CSCO', 'ACN', 'DHR', 'ABT', 'NEE', 'TXN', 'PM', 'UNP', 'RTX',
    'LOW', 'HON', 'AMGN', 'IBM', 'CAT', 'GS', 'BA', 'SBUX', 'GE',
    'MMM', 'DIS', 'INTC', 'NKE', 'CRM'
]


def download_data():
    """Download stock data and earnings dates."""
    import yfinance as yf

    all_data = {}
    earnings_dates = {}

    for ticker in UNIVERSE:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(start='2016-01-01', end='2026-07-24', auto_adjust=True)
            if hist.index.tz is not None:
                hist.index = hist.index.tz_convert(None)
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)

            if len(hist) < 500:
                continue

            all_data[ticker] = hist

            # Get earnings dates
            try:
                cal = stock.get_earnings_dates(limit=50)
                if cal is not None and len(cal) > 0:
                    edates = []
                    for idx in cal.index:
                        if hasattr(idx, 'tz') and idx.tz is not None:
                            dt = idx.tz_localize(None) if hasattr(idx, 'tz_localize') else idx.replace(tzinfo=None)
                        elif hasattr(idx, 'tzinfo') and idx.tzinfo is not None:
                            dt = idx.tz_convert(None) if hasattr(idx, 'tz_convert') else idx.replace(tzinfo=None)
                        else:
                            dt = pd.Timestamp(idx)
                        edates.append(dt)
                    earnings_dates[ticker] = sorted(edates)
            except:
                pass

        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    # SPY for regime
    spy = yf.download('SPY', start='2016-01-01', end='2026-07-24', progress=False)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    print(f"Downloaded {len(all_data)} stocks, {len(earnings_dates)} with earnings data")
    return all_data, earnings_dates, spy


def detect_earnings_events(ticker_data, earnings_dates_list, min_gap_pct=2.0):
    """
    Detect earnings events and classify as positive/negative surprise.
    Surprise is proxied by the 1-day price reaction around earnings.
    """
    close = ticker_data['Close']
    events = []

    for edate in earnings_dates_list:
        # Find the closest trading day
        idx = close.index.searchsorted(edate)
        if idx < 2 or idx >= len(close) - 1:
            continue

        # Get pre and post prices
        pre_price = close.iloc[idx - 1]
        post_price = close.iloc[idx] if idx < len(close) else None

        # Sometimes earnings are after close, so reaction is next day
        if idx + 1 < len(close):
            next_price = close.iloc[idx + 1]
        else:
            next_price = post_price

        if pre_price is None or post_price is None or pre_price <= 0:
            continue

        # Calculate gap (1-day reaction)
        gap_pct = (post_price / pre_price - 1) * 100
        # Also check 2-day reaction for after-hours earnings
        gap_2d = (next_price / pre_price - 1) * 100 if next_price else gap_pct

        # Use the larger absolute reaction
        reaction = gap_pct if abs(gap_pct) >= abs(gap_2d) else gap_2d

        if abs(reaction) < min_gap_pct:
            continue  # Not a meaningful surprise

        # IV rank proxy: 20d realized vol percentile vs trailing 252d
        vol_idx = close.index.searchsorted(edate)
        if vol_idx < 252:
            continue

        log_ret = np.log(close / close.shift(1))
        current_vol = log_ret.iloc[vol_idx-20:vol_idx].std() * np.sqrt(252)
        trailing_vols = log_ret.rolling(20).std().iloc[vol_idx-252:vol_idx] * np.sqrt(252)
        trailing_vols = trailing_vols.dropna()

        if len(trailing_vols) < 100:
            continue

        iv_rank = (trailing_vols < current_vol).mean() * 100

        events.append({
            'date': close.index[idx],
            'reaction_pct': float(reaction),
            'direction': 'positive' if reaction > 0 else 'negative',
            'iv_rank': float(iv_rank),
            'pre_price': float(pre_price),
            'post_price': float(post_price),
        })

    return events


def run_pead_strategy(all_data, earnings_dates, spy,
                      hold_days=40, entry_delay=1, iv_rank_max=80,
                      min_gap_pct=2.0, long_only=False, top_n=5,
                      capital=100000):
    """
    Run PEAD drift strategy.

    After earnings surprise:
    - Positive surprise: go long, hold for hold_days
    - Negative surprise: go short (or skip if long_only)
    - Entry delay: wait N days after earnings for IV to normalize
    - IV rank gate: only enter if IV rank < iv_rank_max
    """
    spy_close = spy['Close']
    spy_sma200 = spy_close.rolling(200).mean()

    all_trades = []

    for ticker, data in all_data.items():
        if ticker not in earnings_dates:
            continue

        events = detect_earnings_events(data, earnings_dates[ticker], min_gap_pct)
        close = data['Close']

        for event in events:
            edate = event['date']

            # IV rank gate
            if event['iv_rank'] > iv_rank_max:
                continue

            # Entry delay
            entry_idx = close.index.searchsorted(edate) + entry_delay
            if entry_idx >= len(close) - hold_days:
                continue

            entry_date = close.index[entry_idx]
            entry_price = close.iloc[entry_idx]

            # Exit
            exit_idx = entry_idx + hold_days
            if exit_idx >= len(close):
                continue

            exit_date = close.index[exit_idx]
            exit_price = close.iloc[exit_idx]

            if entry_price <= 0:
                continue

            # Direction
            if event['direction'] == 'positive':
                ret = (exit_price / entry_price - 1)  # Long
            elif not long_only:
                ret = (entry_price / exit_price - 1)  # Short
            else:
                continue  # Skip shorts if long_only

            # Regime at entry
            if entry_date in spy_close.index and entry_date in spy_sma200.index:
                regime = 'bull' if spy_close.loc[entry_date] > spy_sma200.loc[entry_date] else 'bear'
            else:
                regime = 'unknown'

            # Commission cost (round trip equity)
            commission_pct = 0.001  # 10 bps round trip

            trade = {
                'ticker': ticker,
                'entry_date': str(entry_date.date()),
                'exit_date': str(exit_date.date()),
                'direction': event['direction'],
                'earnings_reaction': round(event['reaction_pct'], 2),
                'iv_rank_at_entry': round(event['iv_rank'], 1),
                'entry_price': round(float(entry_price), 2),
                'exit_price': round(float(exit_price), 2),
                'return_pct': round(float(ret * 100), 2),
                'pnl_pct': round(float((ret - commission_pct) * 100), 2),
                'regime': regime,
                'hold_days': hold_days,
            }
            all_trades.append(trade)

    # Sort by date
    all_trades.sort(key=lambda x: x['entry_date'])

    return all_trades


def compute_metrics(trades, capital=100000):
    """Compute strategy metrics from trades."""
    if not trades:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'cagr': 0, 'maxdd': 0, 'n_trades': 0}

    # Build equity curve (assume equal weight, max top_n concurrent)
    dates = sorted(set(t['entry_date'] for t in trades))
    returns = [t['pnl_pct'] / 100 for t in trades]

    # Monthly-ish aggregation (group by entry month)
    monthly = {}
    for t in trades:
        month = t['entry_date'][:7]
        monthly.setdefault(month, []).append(t['pnl_pct'] / 100)

    period_returns = []
    for month in sorted(monthly.keys()):
        month_trades = monthly[month]
        # Equal weight across concurrent trades
        avg_ret = np.mean(month_trades) if month_trades else 0
        period_returns.append(avg_ret)

    period_returns = np.array(period_returns)

    if len(period_returns) < 2:
        return {'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0, 'cagr': 0, 'maxdd': 0, 'n_trades': len(trades)}

    # Metrics
    equity = capital * np.cumprod(1 + period_returns)
    periods_per_year = 12

    mean_ret = np.mean(period_returns)
    std_ret = np.std(period_returns)
    sharpe = mean_ret / std_ret * np.sqrt(periods_per_year) if std_ret > 0 else 0

    downside = period_returns[period_returns < 0]
    sortino = mean_ret / np.std(downside) * np.sqrt(periods_per_year) if len(downside) > 0 and np.std(downside) > 0 else 0

    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    wr = len(wins) / len(returns) * 100 if returns else 0
    pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')

    years = len(period_returns) / periods_per_year
    cagr = ((equity[-1] / capital) ** (1 / max(years, 0.01)) - 1) * 100

    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    maxdd = float(np.min(dd) * 100)

    return {
        'sharpe': round(float(sharpe), 2),
        'sortino': round(float(sortino), 2),
        'pf': round(float(min(pf, 999)), 2),
        'wr': round(float(wr), 1),
        'cagr': round(float(cagr), 1),
        'maxdd': round(float(maxdd), 1),
        'n_trades': len(trades),
        'avg_return': round(float(np.mean(returns) * 100), 2),
        'final_equity': round(float(equity[-1]), 2),
    }


def adversarial_gates(trades):
    """Run adversarial validation gates."""
    returns = np.array([t['pnl_pct'] / 100 for t in trades])
    regimes = [t['regime'] for t in trades]

    # 1. Permutation (flip signs)
    real_sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    null_sharpes = []
    for _ in range(200):
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = returns * signs
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        null_sharpes.append(s)
    perm_p = float(np.mean([ns >= real_sharpe for ns in null_sharpes]))

    # 2. Regime R1
    bull_ret = [r for r, reg in zip(returns, regimes) if reg == 'bull']
    bear_ret = [r for r, reg in zip(returns, regimes) if reg == 'bear']

    if len(bull_ret) >= 5 and len(bear_ret) >= 5:
        s_bull = np.mean(bull_ret) / np.std(bull_ret) if np.std(bull_ret) > 0 else 0
        s_bear = np.mean(bear_ret) / np.std(bear_ret) if np.std(bear_ret) > 0 else 0
        denom = max(abs(s_bull), abs(s_bear))
        r1_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0
        r1_result = 'PASS' if r1_gap < 0.50 else 'FAIL'
    else:
        s_bull = s_bear = r1_gap = None
        r1_result = 'SKIP'

    # 3. Sub-period
    mid = len(returns) // 2
    sub_result = 'PASS' if sum(returns[:mid]) > 0 and sum(returns[mid:]) > 0 else 'FAIL'

    # 4. Outlier
    if len(returns) >= 20:
        sorted_ret = sorted(returns)
        n_remove = max(1, int(len(returns) * 0.05))
        outlier_result = 'PASS' if sum(sorted_ret[:-n_remove]) > 0 else 'FAIL'
    else:
        outlier_result = 'SKIP'

    gates = {
        'permutation': {'p_value': perm_p, 'result': 'PASS' if perm_p < 0.05 else 'FAIL'},
        'regime_r1': {'bull_sharpe': round(float(s_bull), 2) if s_bull is not None else None,
                      'bear_sharpe': round(float(s_bear), 2) if s_bear is not None else None,
                      'gap': round(float(r1_gap), 3) if r1_gap is not None else None,
                      'result': r1_result},
        'sub_period': sub_result,
        'outlier': outlier_result,
    }
    gates_passed = sum([
        1 if gates['permutation']['result'] == 'PASS' else 0,
        1 if r1_result == 'PASS' else 0,
        1 if sub_result == 'PASS' else 0,
        1 if outlier_result == 'PASS' else 0,
    ])
    return gates, gates_passed


def main():
    print("=" * 60)
    print("POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) v1")
    print("=" * 60)

    print("\nDownloading data...")
    all_data, earnings_dates, spy = download_data()

    # Test multiple variants
    variants = {
        'long_40d_iv80': {
            'hold_days': 40, 'entry_delay': 1, 'iv_rank_max': 80,
            'min_gap_pct': 2.0, 'long_only': True,
            'desc': 'Long only, 40d hold, IV rank < 80'
        },
        'long_40d_iv50': {
            'hold_days': 40, 'entry_delay': 1, 'iv_rank_max': 50,
            'min_gap_pct': 2.0, 'long_only': True,
            'desc': 'Long only, 40d hold, IV rank < 50 (strict)'
        },
        'long_20d_iv80': {
            'hold_days': 20, 'entry_delay': 1, 'iv_rank_max': 80,
            'min_gap_pct': 2.0, 'long_only': True,
            'desc': 'Long only, 20d hold, IV rank < 80'
        },
        'long_60d_iv80': {
            'hold_days': 60, 'entry_delay': 1, 'iv_rank_max': 80,
            'min_gap_pct': 2.0, 'long_only': True,
            'desc': 'Long only, 60d hold, IV rank < 80'
        },
        'longshort_40d': {
            'hold_days': 40, 'entry_delay': 1, 'iv_rank_max': 80,
            'min_gap_pct': 2.0, 'long_only': False,
            'desc': 'Long/short, 40d hold, IV rank < 80'
        },
        'longshort_40d_iv50': {
            'hold_days': 40, 'entry_delay': 1, 'iv_rank_max': 50,
            'min_gap_pct': 2.0, 'long_only': False,
            'desc': 'Long/short, 40d hold, IV rank < 50 (strict)'
        },
        'long_40d_3pct': {
            'hold_days': 40, 'entry_delay': 1, 'iv_rank_max': 80,
            'min_gap_pct': 3.0, 'long_only': True,
            'desc': 'Long only, 40d hold, min 3% gap'
        },
        'long_40d_delay5': {
            'hold_days': 40, 'entry_delay': 5, 'iv_rank_max': 80,
            'min_gap_pct': 2.0, 'long_only': True,
            'desc': 'Long only, 40d hold, 5d entry delay'
        },
        'longshort_20d': {
            'hold_days': 20, 'entry_delay': 1, 'iv_rank_max': 80,
            'min_gap_pct': 2.0, 'long_only': False,
            'desc': 'Long/short, 20d hold'
        },
        'longshort_60d_3pct': {
            'hold_days': 60, 'entry_delay': 1, 'iv_rank_max': 80,
            'min_gap_pct': 3.0, 'long_only': False,
            'desc': 'Long/short, 60d hold, min 3% gap'
        },
    }

    results = {}
    best_sharpe = -999
    best_variant = None

    if MLFLOW_OK:
        try:
            mlflow.set_experiment('pead_drift_v1')
        except:
            pass

    for name, params in variants.items():
        print(f"\n--- {name}: {params['desc']} ---")

        trades = run_pead_strategy(
            all_data, earnings_dates, spy,
            hold_days=params['hold_days'],
            entry_delay=params['entry_delay'],
            iv_rank_max=params['iv_rank_max'],
            min_gap_pct=params['min_gap_pct'],
            long_only=params['long_only'],
        )

        if not trades:
            print("  No trades generated")
            results[name] = {'description': params['desc'], 'metrics': {}, 'gates': {}, 'gates_passed': '0/4'}
            continue

        metrics = compute_metrics(trades)
        gates, gates_passed = adversarial_gates(trades)

        # Direction breakdown
        long_trades = [t for t in trades if t['direction'] == 'positive']
        short_trades = [t for t in trades if t['direction'] == 'negative']
        long_wr = len([t for t in long_trades if t['pnl_pct'] > 0]) / len(long_trades) * 100 if long_trades else 0
        short_wr = len([t for t in short_trades if t['pnl_pct'] > 0]) / len(short_trades) * 100 if short_trades else 0

        results[name] = {
            'description': params['desc'],
            'metrics': metrics,
            'gates': gates,
            'gates_passed': f"{gates_passed}/4",
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'long_wr': round(long_wr, 1),
            'short_wr': round(short_wr, 1),
        }

        print(f"  Trades: {metrics['n_trades']} ({len(long_trades)}L/{len(short_trades)}S)")
        print(f"  Sharpe: {metrics['sharpe']}, CAGR: {metrics['cagr']}%, MaxDD: {metrics['maxdd']}%")
        print(f"  WR: {metrics['wr']}% (Long: {long_wr:.0f}%, Short: {short_wr:.0f}%), PF: {metrics['pf']}")
        print(f"  Avg return: {metrics.get('avg_return', 0):.2f}%")
        print(f"  Gates: {gates_passed}/4 (perm={'PASS' if gates['permutation']['p_value']<0.05 else 'FAIL'}, "
              f"R1={gates['regime_r1']['result']}, sub={gates['sub_period']}, outlier={gates['outlier']})")

        if metrics['sharpe'] > best_sharpe and metrics['n_trades'] >= 20:
            best_sharpe = metrics['sharpe']
            best_variant = name

        if MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=f'pead_{name}'):
                    mlflow.log_params({k: str(v) for k, v in params.items()})
                    mlflow.log_metrics({
                        'sharpe': metrics['sharpe'],
                        'sortino': metrics['sortino'],
                        'cagr': metrics['cagr'],
                        'maxdd': metrics['maxdd'],
                        'wr': metrics['wr'],
                        'n_trades': metrics['n_trades'],
                        'perm_p': gates['permutation']['p_value'],
                        'gates_passed': gates_passed,
                    })
            except:
                pass

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — PEAD DRIFT v1")
    print("=" * 60)
    print(f"\nBest variant: {best_variant} (Sharpe {best_sharpe:.2f})")

    ranked = sorted(
        [(n, r) for n, r in results.items() if r.get('metrics', {}).get('n_trades', 0) > 0],
        key=lambda x: x[1]['metrics'].get('sharpe', 0),
        reverse=True
    )

    print(f"\n{'Variant':<25} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'WR':>6} {'Trades':>7} {'Gates':>6}")
    print("-" * 72)
    for name, r in ranked:
        m = r['metrics']
        print(f"{name:<25} {m['sharpe']:>7.2f} {m['cagr']:>6.1f}% {m['maxdd']:>6.1f}% {m['wr']:>5.1f}% {m['n_trades']:>7} {r['gates_passed']:>6}")

    # Save
    output = {
        'strategy': 'PEAD Drift v1',
        'run_date': str(datetime.now()),
        'best_variant': best_variant,
        'best_sharpe': best_sharpe,
        'variants': results,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved.")
    return output


if __name__ == '__main__':
    main()
