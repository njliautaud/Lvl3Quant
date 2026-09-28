#!/usr/bin/env python3
"""
Aristocrat Momentum (Quality Momentum Rotation) — 6-Test Adversarial Validation

Strategy: Select top 3 quality stocks by 60-day momentum from universe,
equal weight, monthly rebalance.

Universe: AAPL, MSFT, AVGO, JPM, JNJ, PG, KO, PEP
Capital: $645, ~$215 per position, slippage 2bps
Period: 2022-01-01 to 2026-07-31
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Config
UNIVERSE = ['AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP']
START = '2022-01-01'
END = '2026-07-31'
CAPITAL = 645.0
N_TOP = 3
LOOKBACK = 60
SLIPPAGE_BPS = 2
RANDOM_ITERS = 1000
OUTPUT_PATH = '/home/jupiter/Lvl3Quant/data/aristocrat_momentum_adversarial.json'

np.random.seed(42)


def download_data():
    """Download price data for universe + SPY."""
    tickers = UNIVERSE + ['SPY']
    pad_start = (pd.Timestamp(START) - timedelta(days=LOOKBACK * 2)).strftime('%Y-%m-%d')
    data = yf.download(tickers, start=pad_start, end=END, auto_adjust=True, progress=False)
    close = data['Close'][tickers]
    close = close.ffill().dropna()
    return close


def compute_momentum(close_df, lookback, date, tickers):
    """Compute momentum (return over lookback days) for each ticker at a given date."""
    loc = close_df.index.get_loc(date)
    if loc < lookback:
        return {}
    mom = {}
    for t in tickers:
        p_now = close_df[t].iloc[loc]
        p_prev = close_df[t].iloc[loc - lookback]
        if p_prev > 0:
            mom[t] = (p_now / p_prev) - 1.0
    return mom


def get_rebalance_dates(close_df, start, freq='monthly'):
    """Get month-end rebalance dates within the trading period."""
    trading_dates = close_df.loc[start:].index
    if len(trading_dates) == 0:
        return []

    if freq == 'monthly':
        step = 1
    elif freq == 'bi-monthly':
        step = 2
    elif freq == 'quarterly':
        step = 3
    else:
        step = 1

    groups = trading_dates.to_series().groupby([trading_dates.year, trading_dates.month]).last()
    month_ends = groups.values
    rebal = list(month_ends[::step])
    return [pd.Timestamp(d) for d in rebal]


def run_backtest(close_df, start, end, lookback=60, n_top=3, slippage_bps=2,
                 capital=645.0, select_mode='top', rebal_freq='monthly',
                 tickers=None, random_select=False):
    if tickers is None:
        tickers = list(UNIVERSE)

    available_cols = [t for t in tickers if t in close_df.columns]
    spy_cols = available_cols + (['SPY'] if 'SPY' in close_df.columns else [])
    close_sub = close_df[spy_cols].loc[:end].copy()
    rebal_dates = get_rebalance_dates(close_sub, start, freq=rebal_freq)
    if len(rebal_dates) < 2:
        return {'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
                'total_return': 0, 'n_trades': 0, 'daily_returns': [],
                'trade_pnls': [], 'ticker_pnl': {}}

    trading_start = close_sub.loc[start:].index[0]
    trading_dates = close_sub.loc[trading_start:end].index

    positions = {}  # ticker -> shares
    entry_prices = {}  # ticker -> entry price
    cash = capital
    portfolio_values = []
    daily_rets = []
    n_trades = 0
    ticker_pnl = {t: 0.0 for t in available_cols}

    for date in trading_dates:
        if date in rebal_dates:
            # Sell all
            for t, shares in positions.items():
                price = close_sub.loc[date, t]
                sell_price = price * (1 - slippage_bps / 10000)
                pnl = shares * (sell_price - entry_prices.get(t, sell_price))
                ticker_pnl[t] = ticker_pnl.get(t, 0) + pnl
                cash += shares * sell_price
                if shares > 0:
                    n_trades += 1

            # Select
            if random_select:
                selected = list(np.random.choice(available_cols, size=min(n_top, len(available_cols)), replace=False))
            else:
                mom = compute_momentum(close_sub, lookback, date, available_cols)
                if len(mom) < n_top:
                    positions = {}
                    entry_prices = {}
                    portfolio_values.append(cash)
                    if len(portfolio_values) > 1:
                        daily_rets.append((portfolio_values[-1] / portfolio_values[-2]) - 1)
                    continue

                sorted_tickers = sorted(mom.keys(), key=lambda x: mom[x],
                                        reverse=(select_mode == 'top'))
                selected = sorted_tickers[:n_top]

            # Buy equal weight
            positions = {}
            entry_prices = {}
            alloc_per = cash / len(selected)
            total_cost = 0
            for t in selected:
                price = close_sub.loc[date, t]
                buy_price = price * (1 + slippage_bps / 10000)
                shares = alloc_per / buy_price
                positions[t] = shares
                entry_prices[t] = buy_price
                total_cost += shares * buy_price
                n_trades += 1
            cash -= total_cost

        # Portfolio value
        port_val = cash
        for t, shares in positions.items():
            port_val += shares * close_sub.loc[date, t]

        portfolio_values.append(port_val)
        if len(portfolio_values) > 1:
            daily_rets.append((portfolio_values[-1] / portfolio_values[-2]) - 1)

    portfolio_values = np.array(portfolio_values)
    daily_rets = np.array(daily_rets)

    if len(daily_rets) < 10:
        return {'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0, 'mdd': 0,
                'total_return': 0, 'n_trades': 0, 'daily_returns': [],
                'trade_pnls': [], 'ticker_pnl': {}}

    # Metrics
    ann_factor = np.sqrt(252)
    mean_ret = np.mean(daily_rets)
    std_ret = np.std(daily_rets)
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0

    downside = daily_rets[daily_rets < 0]
    down_std = np.std(downside) if len(downside) > 0 else 1e-9
    sortino = mean_ret / down_std * ann_factor

    # Monthly returns for WR/PF
    monthly_rets = []
    chunk_size = 21
    for i in range(0, len(daily_rets), chunk_size):
        chunk = daily_rets[i:i+chunk_size]
        if len(chunk) > 5:
            monthly_rets.append(np.sum(chunk))
    monthly_rets = np.array(monthly_rets)
    wr = np.mean(monthly_rets > 0) * 100 if len(monthly_rets) > 0 else 0

    wins = monthly_rets[monthly_rets > 0]
    losses = monthly_rets[monthly_rets < 0]
    pf = (np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 999

    # MDD
    cummax = np.maximum.accumulate(portfolio_values)
    dd = (portfolio_values - cummax) / cummax
    mdd = np.min(dd) * 100

    total_ret = (portfolio_values[-1] / portfolio_values[0] - 1) * 100

    return {
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'wr': round(float(wr), 2),
        'pf': round(float(min(pf, 999)), 2),
        'mdd': round(float(mdd), 2),
        'total_return': round(float(total_ret), 2),
        'n_trades': int(n_trades),
        'daily_returns': daily_rets.tolist(),
        'trade_pnls': monthly_rets.tolist() if len(monthly_rets) > 0 else [],
        'ticker_pnl': {t: round(float(v), 2) for t, v in ticker_pnl.items()},
    }


def get_spy_regime(close_df):
    spy = close_df['SPY'].copy()
    sma200 = spy.rolling(200).mean()
    regime = pd.Series('bull', index=spy.index)
    regime[spy < sma200] = 'bear'
    return regime


# ═══════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════

def test1_inverse_signal(close_df):
    print("\n[Test 1] Inverse Signal...")
    real = run_backtest(close_df, START, END, select_mode='top')
    inverse = run_backtest(close_df, START, END, select_mode='bottom')

    real_sharpe = real['sharpe']
    inv_sharpe = inverse['sharpe']
    ratio = inv_sharpe / real_sharpe if real_sharpe != 0 else 999

    passed = ratio < 0.50
    print(f"  Real Sharpe: {real_sharpe:.4f}, Inverse Sharpe: {inv_sharpe:.4f}")
    print(f"  Ratio: {ratio:.4f} {'PASS' if passed else 'FAIL'} (threshold < 0.50)")

    return {
        'test': 'inverse_signal',
        'real_sharpe': real_sharpe,
        'inverse_sharpe': inv_sharpe,
        'ratio': round(float(ratio), 4),
        'passed': bool(passed),
        'real_metrics': {k: v for k, v in real.items() if k not in ('daily_returns', 'trade_pnls')},
        'inverse_metrics': {k: v for k, v in inverse.items() if k not in ('daily_returns', 'trade_pnls')},
    }, real


def test2_random_timing(close_df, real_sharpe):
    print("\n[Test 2] Random Timing (1000 iterations)...")
    random_sharpes = []
    for i in range(RANDOM_ITERS):
        if (i + 1) % 200 == 0:
            print(f"  Iteration {i+1}/{RANDOM_ITERS}...")
        res = run_backtest(close_df, START, END, random_select=True)
        random_sharpes.append(res['sharpe'])

    random_sharpes = np.array(random_sharpes)
    percentile = float(np.mean(random_sharpes < real_sharpe) * 100)
    p_value = 1 - percentile / 100

    passed = p_value < 0.05
    print(f"  Real Sharpe: {real_sharpe:.4f}")
    print(f"  Random mean: {np.mean(random_sharpes):.4f}, std: {np.std(random_sharpes):.4f}")
    print(f"  Percentile: {percentile:.1f}%, p-value: {p_value:.4f}")
    print(f"  {'PASS' if passed else 'FAIL'}")

    return {
        'test': 'random_timing',
        'real_sharpe': real_sharpe,
        'random_mean': round(float(np.mean(random_sharpes)), 4),
        'random_std': round(float(np.std(random_sharpes)), 4),
        'random_median': round(float(np.median(random_sharpes)), 4),
        'percentile': round(float(percentile), 2),
        'p_value': round(float(p_value), 4),
        'passed': bool(passed),
    }


def test3_subperiod_stability(close_df):
    print("\n[Test 3] Sub-Period Stability...")
    trading_dates = close_df.loc[START:END].index
    n = len(trading_dates)
    quarter = n // 4
    periods = [
        (trading_dates[0], trading_dates[quarter - 1]),
        (trading_dates[quarter], trading_dates[2 * quarter - 1]),
        (trading_dates[2 * quarter], trading_dates[3 * quarter - 1]),
        (trading_dates[3 * quarter], trading_dates[-1]),
    ]

    sub_results = []
    all_positive = True
    for i, (s, e) in enumerate(periods):
        s_str = s.strftime('%Y-%m-%d')
        e_str = e.strftime('%Y-%m-%d')
        res = run_backtest(close_df, s_str, e_str)
        sub_results.append({
            'period': f"{s_str} to {e_str}",
            'sharpe': res['sharpe'],
            'sortino': res['sortino'],
            'total_return': res['total_return'],
            'mdd': res['mdd'],
        })
        if res['sharpe'] <= 0:
            all_positive = False
        print(f"  Period {i+1} ({s_str} to {e_str}): Sharpe={res['sharpe']:.4f}, Return={res['total_return']:.1f}%")

    passed = all_positive
    print(f"  {'PASS' if passed else 'FAIL'} - All positive: {all_positive}")

    return {
        'test': 'subperiod_stability',
        'sub_periods': sub_results,
        'all_positive_sharpe': bool(all_positive),
        'passed': bool(passed),
    }


def test4_remove_top3_tickers(close_df, real_result):
    print("\n[Test 4] Remove Top 3 Tickers...")
    ticker_pnl = real_result['ticker_pnl']
    sorted_by_pnl = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
    top3 = [t for t, _ in sorted_by_pnl[:3]]
    remaining = [t for t in UNIVERSE if t not in top3]

    print(f"  Top 3 PnL tickers removed: {top3}")
    print(f"  Remaining: {remaining}")

    reduced = run_backtest(close_df, START, END, tickers=remaining,
                           n_top=min(N_TOP, len(remaining)))
    real_sharpe = real_result['sharpe']
    reduced_sharpe = reduced['sharpe']
    drop = 1 - (reduced_sharpe / real_sharpe) if real_sharpe != 0 else 1.0

    passed = drop < 0.50
    print(f"  Real Sharpe: {real_sharpe:.4f}, Reduced Sharpe: {reduced_sharpe:.4f}")
    print(f"  Drop: {drop*100:.1f}% {'PASS' if passed else 'FAIL'} (threshold < 50%)")

    return {
        'test': 'remove_top3_tickers',
        'removed_tickers': top3,
        'remaining_tickers': remaining,
        'real_sharpe': real_sharpe,
        'reduced_sharpe': reduced_sharpe,
        'sharpe_drop_pct': round(float(drop * 100), 2),
        'passed': bool(passed),
        'reduced_metrics': {k: v for k, v in reduced.items() if k not in ('daily_returns', 'trade_pnls')},
    }


def test5_parameter_sensitivity(close_df):
    print("\n[Test 5] Parameter Sensitivity Sweep...")
    lookbacks = [30, 45, 60, 90]
    top_ns = [2, 3, 4]
    freqs = ['monthly', 'bi-monthly', 'quarterly']

    combos = list(product(lookbacks, top_ns, freqs))
    results = []
    above_threshold = 0

    for lb, tn, freq in combos:
        res = run_backtest(close_df, START, END, lookback=lb, n_top=tn, rebal_freq=freq)
        results.append({
            'lookback': lb, 'top_n': tn, 'rebal_freq': freq,
            'sharpe': res['sharpe'], 'total_return': res['total_return'],
        })
        if res['sharpe'] > 0.3:
            above_threshold += 1

    pct_above = above_threshold / len(combos) * 100
    passed = pct_above >= 30

    print(f"  Total combos: {len(combos)}")
    print(f"  Combos with Sharpe > 0.3: {above_threshold} ({pct_above:.1f}%)")

    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Best: lb={results[0]['lookback']}, top={results[0]['top_n']}, "
          f"freq={results[0]['rebal_freq']}, Sharpe={results[0]['sharpe']:.4f}")
    print(f"  Worst: lb={results[-1]['lookback']}, top={results[-1]['top_n']}, "
          f"freq={results[-1]['rebal_freq']}, Sharpe={results[-1]['sharpe']:.4f}")
    print(f"  {'PASS' if passed else 'FAIL'} (threshold >= 30%)")

    return {
        'test': 'parameter_sensitivity',
        'total_combos': len(combos),
        'combos_above_0_3': above_threshold,
        'pct_above_0_3': round(float(pct_above), 2),
        'passed': bool(passed),
        'all_results': results,
        'best_combo': results[0],
        'worst_combo': results[-1],
    }


def test6_cost_sensitivity(close_df):
    print("\n[Test 6] Cost Sensitivity...")
    slippages = [5, 10, 20, 50]
    results = []

    for slip in slippages:
        res = run_backtest(close_df, START, END, slippage_bps=slip)
        results.append({
            'slippage_bps': slip,
            'sharpe': res['sharpe'],
            'total_return': res['total_return'],
        })
        print(f"  {slip} bps: Sharpe={res['sharpe']:.4f}, Return={res['total_return']:.1f}%")

    sharpes = [r['sharpe'] for r in results]
    slips = [r['slippage_bps'] for r in results]

    # If all positive at 50bps, test higher
    if all(s > 0 for s in sharpes):
        for extra_slip in [75, 100, 150, 200, 300, 500]:
            res = run_backtest(close_df, START, END, slippage_bps=extra_slip)
            slips.append(extra_slip)
            sharpes.append(res['sharpe'])
            results.append({
                'slippage_bps': extra_slip,
                'sharpe': res['sharpe'],
                'total_return': res['total_return'],
            })
            if res['sharpe'] <= 0:
                break

    # Interpolate breakeven
    breakeven = None
    for i in range(len(slips) - 1):
        if sharpes[i] > 0 and sharpes[i + 1] <= 0:
            breakeven = slips[i] + (0 - sharpes[i]) / (sharpes[i + 1] - sharpes[i]) * (slips[i + 1] - slips[i])
            break

    if breakeven is None:
        if all(s > 0 for s in sharpes):
            breakeven = float(slips[-1])
        else:
            breakeven = 0.0

    passed = breakeven >= 20
    print(f"  Breakeven slippage: {breakeven:.0f} bps")
    print(f"  {'PASS' if passed else 'FAIL'} (threshold >= 20 bps)")

    return {
        'test': 'cost_sensitivity',
        'slippage_results': results,
        'breakeven_bps': round(float(breakeven), 1),
        'passed': bool(passed),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("ARISTOCRAT MOMENTUM - ADVERSARIAL VALIDATION")
    print("=" * 70)

    print("\nDownloading data...")
    close_df = download_data()
    print(f"Data: {close_df.index[0].strftime('%Y-%m-%d')} to {close_df.index[-1].strftime('%Y-%m-%d')}, "
          f"{len(close_df)} trading days")

    regime = get_spy_regime(close_df)
    regime_trading = regime.loc[START:END]
    bull_pct = float((regime_trading == 'bull').mean() * 100)
    bear_pct = float((regime_trading == 'bear').mean() * 100)
    print(f"Regime split: Bull {bull_pct:.1f}%, Bear {bear_pct:.1f}%")

    print("\nRunning baseline backtest...")
    baseline = run_backtest(close_df, START, END)
    print(f"Baseline: Sharpe={baseline['sharpe']:.4f}, Sortino={baseline['sortino']:.4f}, "
          f"WR={baseline['wr']:.1f}%, PF={baseline['pf']:.2f}, "
          f"MDD={baseline['mdd']:.1f}%, Return={baseline['total_return']:.1f}%, "
          f"Trades={baseline['n_trades']}")

    results = {}

    t1, real_result = test1_inverse_signal(close_df)
    results['test1_inverse_signal'] = t1

    results['test2_random_timing'] = test2_random_timing(close_df, real_result['sharpe'])
    results['test3_subperiod_stability'] = test3_subperiod_stability(close_df)
    results['test4_remove_top3'] = test4_remove_top3_tickers(close_df, real_result)
    results['test5_parameter_sensitivity'] = test5_parameter_sensitivity(close_df)
    results['test6_cost_sensitivity'] = test6_cost_sensitivity(close_df)

    n_pass = sum(1 for v in results.values() if v.get('passed', False))
    n_total = len(results)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass}/{n_total} tests passed")
    print("=" * 70)
    for name, res in results.items():
        status = "PASS" if res['passed'] else "FAIL"
        print(f"  [{status}] {res['test']}")

    output = {
        'strategy': 'Aristocrat Momentum (Quality Momentum Rotation)',
        'universe': UNIVERSE,
        'period': f"{START} to {END}",
        'capital': CAPITAL,
        'slippage_bps': SLIPPAGE_BPS,
        'timestamp': datetime.now().isoformat(),
        'baseline': {
            'sharpe': baseline['sharpe'],
            'sortino': baseline['sortino'],
            'wr': baseline['wr'],
            'pf': baseline['pf'],
            'mdd': baseline['mdd'],
            'total_return': baseline['total_return'],
            'n_trades': baseline['n_trades'],
            'ticker_pnl': baseline['ticker_pnl'],
        },
        'regime': {
            'bull_pct': round(bull_pct, 2),
            'bear_pct': round(bear_pct, 2),
        },
        'tests': results,
        'summary': {
            'tests_passed': n_pass,
            'tests_total': n_total,
            'pass_rate': round(n_pass / n_total * 100, 1),
            'overall_pass': n_pass >= 4,
        },
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
