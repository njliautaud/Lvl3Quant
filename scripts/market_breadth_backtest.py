#!/usr/bin/env python3
"""
Market Breadth Divergence Strategy Backtest
============================================
Academic basis: Zweig (1986) "Winning on Wall Street"
- Breadth divergences (market up, fewer stocks participating) predict corrections
- Breadth thrusts (sudden broad participation) predict rallies

Uses ETF proxies:
- RSP/SPY ratio for breadth participation
- 11 sector ETFs above/below 50-SMA for sector breadth

Variants A-F tested with 5-gate validation.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
SLIPPAGE_PCT = 0.0002  # 0.02%
STARTING_CAPITAL = 645.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
DOWNLOAD_START = '2020-01-01'  # extra history for SMA warmup

SECTOR_ETFS = ['XLK', 'XLC', 'XLY', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLP']
TICKERS = ['QQQ', 'SPY', 'RSP'] + SECTOR_ETFS

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/market_breadth_results.json'


def download_data():
    """Download all required ETF data."""
    print("Downloading ETF data...")
    data = {}
    for ticker in TICKERS:
        df = yf.download(ticker, start=DOWNLOAD_START, end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df['Close'].copy()

    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.dropna()
    print(f"  Got {len(prices)} trading days from {prices.index[0].date()} to {prices.index[-1].date()}")
    return prices


def compute_features(prices):
    """Compute all breadth features."""
    feat = pd.DataFrame(index=prices.index)

    # RSP/SPY ratio and its 20-day change
    feat['rsp_spy_ratio'] = prices['RSP'] / prices['SPY']
    feat['rsp_spy_20d_chg'] = feat['rsp_spy_ratio'].pct_change(20)

    # Sector breadth: count sectors above their own 50-SMA
    sector_above_50sma = pd.DataFrame(index=prices.index)
    for s in SECTOR_ETFS:
        sma50 = prices[s].rolling(50).mean()
        sector_above_50sma[s] = (prices[s] > sma50).astype(int)
    feat['sectors_above_50sma'] = sector_above_50sma.sum(axis=1)

    # SPY 20-day high (for divergence detection)
    feat['spy_20d_high'] = prices['SPY'].rolling(20).max()
    feat['spy_at_20d_high'] = (prices['SPY'] >= feat['spy_20d_high'] * 0.999)  # within 0.1%

    # RSP/SPY ratio 20-day low (for divergence detection)
    feat['rsp_spy_20d_low'] = feat['rsp_spy_ratio'].rolling(20).min()
    feat['rsp_spy_at_20d_low'] = (feat['rsp_spy_ratio'] <= feat['rsp_spy_20d_low'] * 1.001)

    # QQQ returns for benchmarking
    feat['qqq_ret'] = prices['QQQ'].pct_change()

    return feat


def run_backtest(prices, signals, variant_name):
    """
    Run a long/cash backtest on QQQ given a boolean signal series.
    signal=True → long QQQ, signal=False → cash.
    Returns equity curve, trades, and metrics.
    """
    # Restrict to OOT period
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    qqq = prices['QQQ'][oot_mask].copy()
    sig = signals[oot_mask].copy()

    # Align
    common_idx = qqq.index.intersection(sig.index)
    qqq = qqq.loc[common_idx]
    sig = sig.loc[common_idx]

    # Drop NaN signals
    valid = sig.notna()
    qqq = qqq[valid]
    sig = sig[valid]

    if len(qqq) < 50:
        return None

    # Simulate
    cash = STARTING_CAPITAL
    shares = 0.0
    equity = []
    trades = []
    in_trade = False
    entry_price = 0
    entry_date = None

    for i in range(len(qqq)):
        price = qqq.iloc[i]
        date = qqq.index[i]
        want_long = bool(sig.iloc[i])

        if want_long and not in_trade:
            # Enter long
            exec_price = price * (1 + SLIPPAGE_PCT)
            shares = cash / exec_price
            cash = 0
            in_trade = True
            entry_price = exec_price
            entry_date = date
        elif not want_long and in_trade:
            # Exit to cash
            exec_price = price * (1 - SLIPPAGE_PCT)
            cash = shares * exec_price
            pnl_pct = (exec_price / entry_price - 1) * 100
            trades.append({
                'entry_date': str(entry_date.date()),
                'exit_date': str(date.date()),
                'entry_price': round(entry_price, 2),
                'exit_price': round(exec_price, 2),
                'pnl_pct': round(pnl_pct, 2),
                'holding_days': (date - entry_date).days
            })
            shares = 0
            in_trade = False

        # Mark to market
        if in_trade:
            equity.append(shares * price)
        else:
            equity.append(cash)

    # Close any open trade at end
    if in_trade:
        price = qqq.iloc[-1]
        exec_price = price * (1 - SLIPPAGE_PCT)
        cash = shares * exec_price
        pnl_pct = (exec_price / entry_price - 1) * 100
        trades.append({
            'entry_date': str(entry_date.date()),
            'exit_date': str(qqq.index[-1].date()),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exec_price, 2),
            'pnl_pct': round(pnl_pct, 2),
            'holding_days': (qqq.index[-1] - entry_date).days
        })

    equity_series = pd.Series(equity, index=qqq.index)
    return {
        'equity': equity_series,
        'trades': trades,
        'final_equity': equity_series.iloc[-1],
    }


def compute_metrics(result, prices):
    """Compute performance metrics from backtest result."""
    eq = result['equity']
    trades = result['trades']

    daily_ret = eq.pct_change().dropna()

    # Annualized metrics
    n_years = len(daily_ret) / 252
    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe (annualized, rf=0)
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_dd = drawdown.min()

    # Win rate
    winning = [t for t in trades if t['pnl_pct'] > 0]
    wr = len(winning) / len(trades) * 100 if trades else 0

    # Profit factor
    gross_profit = sum(t['pnl_pct'] for t in trades if t['pnl_pct'] > 0)
    gross_loss = abs(sum(t['pnl_pct'] for t in trades if t['pnl_pct'] < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Buy & hold QQQ benchmark
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    qqq_oot = prices['QQQ'][oot_mask].dropna()
    bh_ret = (qqq_oot.iloc[-1] / qqq_oot.iloc[0]) - 1
    bh_daily = qqq_oot.pct_change().dropna()
    bh_sharpe = (bh_daily.mean() / bh_daily.std()) * np.sqrt(252) if bh_daily.std() > 0 else 0
    bh_cummax = qqq_oot.cummax()
    bh_dd = ((qqq_oot - bh_cummax) / bh_cummax).min()

    # Avg holding days
    avg_hold = np.mean([t['holding_days'] for t in trades]) if trades else 0

    # Time in market
    eq_daily_ret = eq.pct_change().dropna()
    # Approximate: days where equity moved with market
    in_market_days = sum(1 for r in eq_daily_ret if abs(r) > 1e-8)
    time_in_market = in_market_days / len(eq_daily_ret) * 100 if len(eq_daily_ret) > 0 else 0

    return {
        'total_return_pct': round(total_ret * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'profit_factor': round(pf, 3),
        'win_rate_pct': round(wr, 1),
        'num_trades': len(trades),
        'avg_holding_days': round(avg_hold, 1),
        'time_in_market_pct': round(time_in_market, 1),
        'final_equity': round(eq.iloc[-1], 2),
        'starting_equity': STARTING_CAPITAL,
        'benchmark_buy_hold_return_pct': round(bh_ret * 100, 2),
        'benchmark_sharpe': round(bh_sharpe, 3),
        'benchmark_max_dd_pct': round(bh_dd * 100, 2),
    }


def permutation_test(result, prices, n_perms=1000):
    """Permutation test: shuffle signal timing, compare Sharpe."""
    eq = result['equity']
    daily_ret = eq.pct_change().dropna()
    actual_sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() > 0 else 0

    # Get QQQ returns in OOT period
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    qqq_ret = prices['QQQ'][oot_mask].pct_change().dropna()

    # For permutation: randomly decide in/out of market each day
    rng = np.random.RandomState(42)
    time_in = len(daily_ret[daily_ret.abs() > 1e-8]) / len(daily_ret)

    count_better = 0
    for _ in range(n_perms):
        # Random signal with same time-in-market
        random_in = rng.random(len(qqq_ret)) < time_in
        perm_ret = qqq_ret.values.copy()
        perm_ret[~random_in] = 0
        if perm_ret.std() > 0:
            perm_sharpe = (perm_ret.mean() / perm_ret.std()) * np.sqrt(252)
        else:
            perm_sharpe = 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return p_value


def regime_analysis(result, prices):
    """Analyze performance in up vs down market regimes."""
    eq = result['equity']

    # Define regimes: SPY monthly return > 0 = bull, < 0 = bear
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    spy_oot = prices['SPY'][oot_mask].dropna()

    # Monthly regime classification
    spy_monthly = spy_oot.resample('ME').last()
    spy_monthly_ret = spy_monthly.pct_change()

    # Map each day to its month's regime
    eq_daily_ret = eq.pct_change().dropna()

    bull_rets = []
    bear_rets = []

    for date, ret in eq_daily_ret.items():
        # Find the month
        month_end = date + pd.offsets.MonthEnd(0)
        if month_end in spy_monthly_ret.index:
            if spy_monthly_ret[month_end] > 0:
                bull_rets.append(ret)
            else:
                bear_rets.append(ret)

    bull_rets = np.array(bull_rets)
    bear_rets = np.array(bear_rets)

    bull_sharpe = (bull_rets.mean() / bull_rets.std() * np.sqrt(252)) if len(bull_rets) > 10 and bull_rets.std() > 0 else 0
    bear_sharpe = (bear_rets.mean() / bear_rets.std() * np.sqrt(252)) if len(bear_rets) > 10 and bear_rets.std() > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0

    return {
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'bull_days': len(bull_rets),
        'bear_days': len(bear_rets),
    }


def five_gate_validation(metrics, p_value, regime):
    """Apply 5-gate validation."""
    gates = {}
    gates['1_sharpe_gt_0.5'] = metrics['sharpe'] > 0.5
    gates['2_perm_p_lt_0.05'] = p_value < 0.05
    gates['3_regime_gap_lt_0.5'] = regime['regime_gap'] < 0.5
    gates['4_maxdd_gt_neg50'] = metrics['max_drawdown_pct'] > -50
    gates['5_min_20_trades'] = metrics['num_trades'] >= 20
    gates['all_passed'] = all(gates.values())
    return gates


def generate_signals(prices, feat):
    """Generate signal series for each variant."""
    signals = {}

    # A: RSP/SPY ratio — long QQQ when RSP/SPY 20d change > 0
    signals['A_RSP_SPY_Ratio'] = feat['rsp_spy_20d_chg'] > 0

    # B: Sector breadth — long QQQ when ≥8/11 sectors above 50-SMA
    signals['B_Sector_Breadth'] = feat['sectors_above_50sma'] >= 8

    # C: Combined — long when BOTH RSP outperforming AND ≥6/11 sectors above 50-SMA
    signals['C_Combined'] = (feat['rsp_spy_20d_chg'] > 0) & (feat['sectors_above_50sma'] >= 6)

    # D: Breadth thrust — long when ≥10/11 sectors above 50-SMA (rare, high-conviction)
    signals['D_Breadth_Thrust'] = feat['sectors_above_50sma'] >= 10

    # E: Breadth divergence — CASH when SPY at 20d high but RSP/SPY ratio at 20d low
    # Default is long, go to cash on divergence
    divergence = feat['spy_at_20d_high'] & feat['rsp_spy_at_20d_low']
    # Stay in cash for 10 days after divergence detected
    div_cash = divergence.rolling(10).max().fillna(0).astype(bool)
    signals['E_Divergence_Filter'] = ~div_cash

    # F: Breadth score composite (0-3)
    score = pd.Series(0, index=feat.index)
    score += (feat['rsp_spy_20d_chg'] > 0).astype(int)
    score += (feat['sectors_above_50sma'] >= 6).astype(int)
    score += (feat['sectors_above_50sma'] >= 8).astype(int)
    signals['F_Breadth_Score'] = score >= 2

    return signals


def main():
    print("=" * 70)
    print("MARKET BREADTH DIVERGENCE STRATEGY BACKTEST")
    print("Zweig (1986) — Breadth Divergence & Thrust Signals")
    print(f"OOT Period: {OOT_START} to {OOT_END}")
    print(f"Starting Capital: ${STARTING_CAPITAL}")
    print("=" * 70)

    # Download data
    prices = download_data()

    # Compute features
    feat = compute_features(prices)

    # Generate signals
    signals = generate_signals(prices, feat)

    # Run backtests
    all_results = {}

    for name, sig in signals.items():
        print(f"\n{'─' * 50}")
        print(f"Variant {name}")
        print(f"{'─' * 50}")

        result = run_backtest(prices, sig, name)
        if result is None:
            print(f"  SKIPPED — insufficient data")
            continue

        metrics = compute_metrics(result, prices)
        p_value = permutation_test(result, prices)
        regime = regime_analysis(result, prices)
        gates = five_gate_validation(metrics, p_value, regime)

        # Print summary
        print(f"  Total Return: {metrics['total_return_pct']:+.1f}% | CAGR: {metrics['cagr_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f}")
        print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}% | Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Win Rate: {metrics['win_rate_pct']:.0f}% | Trades: {metrics['num_trades']}")
        print(f"  Avg Hold: {metrics['avg_holding_days']:.0f} days | Time in Market: {metrics['time_in_market_pct']:.0f}%")
        print(f"  Final Equity: ${metrics['final_equity']:.2f}")
        print(f"  Benchmark (B&H QQQ): {metrics['benchmark_buy_hold_return_pct']:+.1f}% | Sharpe: {metrics['benchmark_sharpe']:.3f}")
        print(f"  Permutation p-value: {p_value:.3f}")
        print(f"  Regime — Bull Sharpe: {regime['bull_sharpe']:.3f}, Bear Sharpe: {regime['bear_sharpe']:.3f}, Gap: {regime['regime_gap']:.3f}")
        print(f"  5-Gate: {'PASS ✓' if gates['all_passed'] else 'FAIL ✗'}")
        for g, v in gates.items():
            if g != 'all_passed':
                print(f"    {g}: {'✓' if v else '✗'}")

        all_results[name] = {
            'metrics': metrics,
            'permutation_p_value': round(p_value, 4),
            'regime_analysis': regime,
            'five_gate_validation': gates,
            'trades': result['trades'],
        }

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>7} {'WR':>5} {'Trades':>7} {'5-Gate':>7}")
    print(f"{'─' * 25} {'─' * 7} {'─' * 8} {'─' * 8} {'─' * 7} {'─' * 5} {'─' * 7} {'─' * 7}")

    for name, r in all_results.items():
        m = r['metrics']
        passed = 'PASS' if r['five_gate_validation']['all_passed'] else 'FAIL'
        print(f"{name:<25} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>+7.1f}% {m['max_drawdown_pct']:>6.1f}% {m['win_rate_pct']:>4.0f}% {m['num_trades']:>7} {passed:>7}")

    # Save results
    output = {
        'strategy': 'Market Breadth Divergence',
        'academic_basis': 'Zweig (1986) Winning on Wall Street',
        'oot_period': f'{OOT_START} to {OOT_END}',
        'starting_capital': STARTING_CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'run_date': datetime.now().isoformat(),
        'variants': all_results,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")

    # Identify best variant
    passing = {k: v for k, v in all_results.items() if v['five_gate_validation']['all_passed']}
    if passing:
        best = max(passing.items(), key=lambda x: x[1]['metrics']['sharpe'])
        print(f"\nBEST PASSING VARIANT: {best[0]} (Sharpe={best[1]['metrics']['sharpe']:.3f})")
    else:
        print("\nNO VARIANT PASSED ALL 5 GATES")
        # Show which came closest
        best = max(all_results.items(), key=lambda x: sum(v for k, v in x[1]['five_gate_validation'].items() if k != 'all_passed'))
        gates_passed = sum(v for k, v in best[1]['five_gate_validation'].items() if k != 'all_passed')
        print(f"Closest: {best[0]} ({gates_passed}/5 gates, Sharpe={best[1]['metrics']['sharpe']:.3f})")


if __name__ == '__main__':
    main()
