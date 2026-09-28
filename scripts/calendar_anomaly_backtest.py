#!/usr/bin/env python3
"""
Calendar/Seasonal Anomaly Backtest
Tests 6 variants (A-F) of well-documented calendar effects.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ==============================================================================
# CONFIG
# ==============================================================================
TICKERS = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'TLT']
START = '2022-01-01'
END = '2026-07-28'
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per trade (applied on entry and exit)
COMMISSION = 0.0

# Validation gates
SHARPE_GATE = 0.5
PERM_PVALUE_GATE = 0.05
REGIME_GAP_GATE = 0.5
MAX_DD_GATE = -0.50
MIN_TRADES_GATE = 20
N_PERMUTATIONS = 1000

# ==============================================================================
# DATA DOWNLOAD
# ==============================================================================
print("Downloading data...")
data = {}
for t in TICKERS:
    df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[t] = df
    print(f"  {t}: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

spy = data['SPY'].copy()
spy_close = spy['Close']

# 200-day SMA for regime classification
spy['SMA200'] = spy_close.rolling(200).mean()
spy['regime'] = np.where(spy_close > spy['SMA200'], 'bull', 'bear')

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def compute_metrics(trades_df, capital=CAPITAL):
    """Compute performance metrics from a trades DataFrame.
    trades_df must have columns: entry_date, exit_date, entry_price, exit_price, ticker
    """
    if len(trades_df) == 0:
        return _empty_metrics()

    # Apply slippage
    trades_df = trades_df.copy()
    trades_df['entry_cost'] = trades_df['entry_price'] * (1 + SLIPPAGE_PCT)
    trades_df['exit_proceeds'] = trades_df['exit_price'] * (1 - SLIPPAGE_PCT)

    # Calculate returns per trade
    trades_df['return'] = (trades_df['exit_proceeds'] / trades_df['entry_cost']) - 1

    # Simulate equity curve (fully invested per trade, sequential)
    equity = capital
    equity_curve = [capital]
    daily_returns = []

    for _, trade in trades_df.iterrows():
        shares = int(equity / trade['entry_cost'])
        if shares == 0:
            shares = 1  # fractional for small accounts
        pnl = shares * (trade['exit_proceeds'] - trade['entry_cost'])
        equity += pnl
        equity_curve.append(equity)
        daily_returns.append(trade['return'])

    daily_returns = np.array(daily_returns)
    equity_curve = np.array(equity_curve)

    # Core metrics
    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    n_trades = len(trades_df)

    # Annualized metrics
    first_date = trades_df['entry_date'].min()
    last_date = trades_df['exit_date'].max()
    years = max((last_date - first_date).days / 365.25, 0.01)
    cagr = (equity_curve[-1] / equity_curve[0]) ** (1 / years) - 1

    # Sharpe (annualized from trade returns)
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        trades_per_year = n_trades / years
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 1:
        downside_std = downside.std()
        if downside_std > 0:
            trades_per_year = n_trades / years
            sortino = (daily_returns.mean() / downside_std) * np.sqrt(trades_per_year)
        else:
            sortino = np.inf if daily_returns.mean() > 0 else 0.0
    else:
        sortino = np.inf if daily_returns.mean() > 0 else 0.0

    # Win rate
    win_rate = (daily_returns > 0).sum() / len(daily_returns)

    # Profit factor
    gross_profit = daily_returns[daily_returns > 0].sum()
    gross_loss = abs(daily_returns[daily_returns < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = dd.min()

    return {
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'total_return': round(float(total_return), 4),
        'max_drawdown': round(float(max_dd), 4),
        'n_trades': int(n_trades),
        'profit_factor': round(float(min(profit_factor, 99.99)), 4),
        'win_rate': round(float(win_rate), 4),
        'cagr': round(float(cagr), 4),
        'final_equity': round(float(equity_curve[-1]), 2),
        'daily_returns': daily_returns,
        'trades_df': trades_df
    }


def _empty_metrics():
    return {
        'sharpe': 0.0, 'sortino': 0.0, 'total_return': 0.0,
        'max_drawdown': 0.0, 'n_trades': 0, 'profit_factor': 0.0,
        'win_rate': 0.0, 'cagr': 0.0, 'final_equity': CAPITAL,
        'daily_returns': np.array([]), 'trades_df': pd.DataFrame()
    }


def regime_analysis(trades_df):
    """Split trades by bull/bear regime and compute regime-specific Sharpe."""
    if len(trades_df) == 0:
        return {'bull_sharpe': 0.0, 'bear_sharpe': 0.0, 'regime_gap': 0.0,
                'bull_trades': 0, 'bear_trades': 0}

    trades_df = trades_df.copy()
    # Classify each trade by regime at entry
    regimes = []
    for _, t in trades_df.iterrows():
        entry = t['entry_date']
        # Find closest date in spy
        idx = spy.index.get_indexer([entry], method='ffill')[0]
        if idx >= 0 and idx < len(spy):
            regimes.append(spy['regime'].iloc[idx])
        else:
            regimes.append('unknown')
    trades_df['regime'] = regimes

    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']

    bull_metrics = compute_metrics(bull) if len(bull) > 0 else _empty_metrics()
    bear_metrics = compute_metrics(bear) if len(bear) > 0 else _empty_metrics()

    bull_sharpe = bull_metrics['sharpe']
    bear_sharpe = bear_metrics['sharpe']

    denom = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom if denom > 0 else 0.0

    return {
        'bull_sharpe': round(float(bull_sharpe), 4),
        'bear_sharpe': round(float(bear_sharpe), 4),
        'regime_gap': round(float(regime_gap), 4),
        'bull_trades': int(len(bull)),
        'bear_trades': int(len(bear))
    }


def permutation_test(trades_df, original_sharpe, n_perms=N_PERMUTATIONS):
    """Random-entry permutation test: pick random entry dates from SPY, same hold periods, compute Sharpe."""
    if len(trades_df) < 5:
        return 1.0

    trades = trades_df.copy()
    # Compute holding periods in trading days
    hold_days = []
    for _, t in trades.iterrows():
        entry_idx = spy_close.index.get_indexer([t['entry_date']], method='ffill')[0]
        exit_idx = spy_close.index.get_indexer([t['exit_date']], method='ffill')[0]
        hold_days.append(max(1, exit_idx - entry_idx))

    n_trades = len(trades)
    spy_prices = spy_close.values
    spy_dates = spy_close.index
    max_idx = len(spy_prices) - 1

    count_above = 0
    for _ in range(n_perms):
        perm_returns = []
        for hd in hold_days:
            entry_idx = np.random.randint(0, max(1, max_idx - hd))
            exit_idx = min(entry_idx + hd, max_idx)
            entry_p = spy_prices[entry_idx] * (1 + SLIPPAGE_PCT)
            exit_p = spy_prices[exit_idx] * (1 - SLIPPAGE_PCT)
            perm_returns.append(exit_p / entry_p - 1)

        perm_returns = np.array(perm_returns)
        if perm_returns.std() > 0:
            # Annualize same way as original
            years = max((spy_dates[-1] - spy_dates[0]).days / 365.25, 0.01)
            trades_per_year = n_trades / years
            shuf_sharpe = (perm_returns.mean() / perm_returns.std()) * np.sqrt(trades_per_year)
        else:
            shuf_sharpe = 0.0
        if shuf_sharpe >= original_sharpe:
            count_above += 1

    return round(count_above / n_perms, 4)


def validate_gates(metrics, regime, perm_pvalue):
    """Check all 5 validation gates."""
    gates = {}
    gates['sharpe_gt_0.5'] = metrics['sharpe'] > SHARPE_GATE
    gates['perm_p_lt_0.05'] = perm_pvalue < PERM_PVALUE_GATE
    gates['regime_gap_lt_0.5'] = regime['regime_gap'] < REGIME_GAP_GATE
    gates['max_dd_gt_neg50'] = metrics['max_drawdown'] > MAX_DD_GATE
    gates['min_20_trades'] = metrics['n_trades'] >= MIN_TRADES_GATE

    n_passed = sum(gates.values())
    verdict = 'PASS' if n_passed == 5 else f'FAIL ({n_passed}/5)'

    return gates, n_passed, verdict


# ==============================================================================
# STRATEGY A: Turn-of-Month (TOM)
# ==============================================================================
def strategy_tom():
    """Buy SPY on last trading day of month, sell on 3rd trading day of next month."""
    print("\n[A] Turn-of-Month (TOM)...")
    prices = spy_close
    dates = prices.index

    trades = []
    # Group by year-month
    months = dates.to_period('M').unique()

    for i in range(len(months) - 1):
        cur_month = months[i]
        next_month = months[i + 1]

        # Last trading day of current month
        cur_month_dates = dates[dates.to_period('M') == cur_month]
        if len(cur_month_dates) == 0:
            continue
        entry_date = cur_month_dates[-1]  # last day

        # 3rd trading day of next month
        next_month_dates = dates[dates.to_period('M') == next_month]
        if len(next_month_dates) < 3:
            continue
        exit_date = next_month_dates[2]  # 3rd day (0-indexed: day 3)

        entry_price = float(prices.loc[entry_date])
        exit_price = float(prices.loc[exit_date])

        trades.append({
            'entry_date': entry_date, 'exit_date': exit_date,
            'entry_price': entry_price, 'exit_price': exit_price,
            'ticker': 'SPY'
        })

    return pd.DataFrame(trades)


# ==============================================================================
# STRATEGY B: Halloween Effect (Sell in May)
# ==============================================================================
def strategy_halloween():
    """Long SPY Nov 1 to Apr 30, cash May-Oct."""
    print("[B] Halloween Effect (Sell in May)...")
    prices = spy_close
    dates = prices.index

    trades = []
    years = sorted(dates.year.unique())

    for y in years:
        # Find entry: first trading day on or after Nov 1
        nov_dates = dates[(dates.year == y) & (dates.month >= 11)]
        if len(nov_dates) == 0:
            continue
        entry_date = nov_dates[0]

        # Find exit: last trading day on or before Apr 30 of next year
        apr_dates = dates[(dates.year == y + 1) & (dates.month <= 4)]
        if len(apr_dates) == 0:
            continue
        exit_date = apr_dates[-1]

        entry_price = float(prices.loc[entry_date])
        exit_price = float(prices.loc[exit_date])

        trades.append({
            'entry_date': entry_date, 'exit_date': exit_date,
            'entry_price': entry_price, 'exit_price': exit_price,
            'ticker': 'SPY'
        })

    return pd.DataFrame(trades)


# ==============================================================================
# STRATEGY C: Monthly Seasonality Rotation
# ==============================================================================
def strategy_monthly_rotation():
    """Each month, buy 2 sector ETFs with best historical avg return for that calendar month."""
    print("[C] Monthly Seasonality Rotation...")
    sector_tickers = ['XLF', 'XLE', 'TLT', 'IWM', 'QQQ', 'SPY']

    # Compute monthly returns for each ticker
    monthly_returns = {}
    for t in sector_tickers:
        prices = data[t]['Close']
        monthly = prices.resample('ME').last().pct_change()
        monthly_returns[t] = monthly

    trades = []
    all_dates = spy_close.index
    months = all_dates.to_period('M').unique()

    for period in months:
        cal_month = period.month

        # Look back: average return for this calendar month over trailing data
        scores = {}
        for t in sector_tickers:
            mr = monthly_returns[t]
            # Filter to same calendar month, before current period
            historical = mr[(mr.index.month == cal_month) & (mr.index.to_period('M') < period)]
            if len(historical) >= 1:
                scores[t] = historical.mean()

        if len(scores) < 2:
            continue

        # Pick top 2
        top2 = sorted(scores, key=scores.get, reverse=True)[:2]

        # Entry: first trading day of month
        month_dates = all_dates[all_dates.to_period('M') == period]
        if len(month_dates) < 2:
            continue
        entry_date = month_dates[0]
        exit_date = month_dates[-1]

        for t in top2:
            t_prices = data[t]['Close']
            if entry_date in t_prices.index and exit_date in t_prices.index:
                trades.append({
                    'entry_date': entry_date, 'exit_date': exit_date,
                    'entry_price': float(t_prices.loc[entry_date]),
                    'exit_price': float(t_prices.loc[exit_date]),
                    'ticker': t
                })

    return pd.DataFrame(trades)


# ==============================================================================
# STRATEGY D: Friday-Monday Effect
# ==============================================================================
def strategy_friday_monday():
    """Buy SPY at Friday close, sell Monday close."""
    print("[D] Friday-Monday Effect...")
    prices = spy_close
    dates = prices.index

    trades = []
    for i in range(len(dates) - 5):
        d = dates[i]
        if d.dayofweek == 4:  # Friday
            # Find next Monday
            for j in range(i + 1, min(i + 5, len(dates))):
                if dates[j].dayofweek == 0:  # Monday
                    entry_date = d
                    exit_date = dates[j]
                    trades.append({
                        'entry_date': entry_date, 'exit_date': exit_date,
                        'entry_price': float(prices.iloc[i]),
                        'exit_price': float(prices.iloc[j]),
                        'ticker': 'SPY'
                    })
                    break

    return pd.DataFrame(trades)


# ==============================================================================
# STRATEGY E: Quadruple Witching Week
# ==============================================================================
def strategy_quad_witching():
    """Buy QQQ 5 days before quarterly expiration (3rd Friday of Mar/Jun/Sep/Dec), sell on expiration."""
    print("[E] Quadruple Witching Week...")
    qqq_prices = data['QQQ']['Close']
    dates = qqq_prices.index

    trades = []
    # Find 3rd Fridays of Mar, Jun, Sep, Dec
    for year in sorted(dates.year.unique()):
        for month in [3, 6, 9, 12]:
            # Find 3rd Friday
            month_dates = dates[(dates.year == year) & (dates.month == month)]
            fridays = month_dates[month_dates.dayofweek == 4]
            if len(fridays) < 3:
                continue
            exp_date = fridays[2]  # 3rd Friday

            # Entry: 5 trading days before
            exp_idx = dates.get_loc(exp_date)
            entry_idx = max(0, exp_idx - 5)
            entry_date = dates[entry_idx]

            if entry_date in qqq_prices.index and exp_date in qqq_prices.index:
                trades.append({
                    'entry_date': entry_date, 'exit_date': exp_date,
                    'entry_price': float(qqq_prices.loc[entry_date]),
                    'exit_price': float(qqq_prices.loc[exp_date]),
                    'ticker': 'QQQ'
                })

    return pd.DataFrame(trades)


# ==============================================================================
# STRATEGY F: ADVERSARIAL - Random Calendar
# ==============================================================================
def strategy_random(reference_trades_df):
    """Buy SPY at random dates, same avg holding period and number of trades as best variant."""
    print("[F] Random Calendar (adversarial)...")
    prices = spy_close
    dates = prices.index

    n_trades = len(reference_trades_df)
    if n_trades == 0:
        return pd.DataFrame()

    # Average holding period from reference
    ref = reference_trades_df.copy()
    avg_hold = int((ref['exit_date'] - ref['entry_date']).dt.days.mean())
    avg_hold = max(avg_hold, 1)

    trades = []
    available_dates = list(range(len(dates) - avg_hold - 5))
    chosen = np.random.choice(available_dates, size=min(n_trades, len(available_dates)), replace=False)
    chosen = sorted(chosen)

    for idx in chosen:
        entry_date = dates[idx]
        exit_idx = min(idx + avg_hold, len(dates) - 1)
        # Find closest actual trading day
        exit_date = dates[exit_idx]

        trades.append({
            'entry_date': entry_date, 'exit_date': exit_date,
            'entry_price': float(prices.iloc[idx]),
            'exit_price': float(prices.iloc[exit_idx]),
            'ticker': 'SPY'
        })

    return pd.DataFrame(trades)


# ==============================================================================
# RUN ALL STRATEGIES
# ==============================================================================
print("\n" + "="*70)
print("CALENDAR ANOMALY BACKTEST")
print(f"Period: {START} to {END} | Capital: ${CAPITAL}")
print("="*70)

strategies = {
    'A_TurnOfMonth': strategy_tom,
    'B_HalloweenEffect': strategy_halloween,
    'C_MonthlyRotation': strategy_monthly_rotation,
    'D_FridayMonday': strategy_friday_monday,
    'E_QuadWitching': strategy_quad_witching,
}

all_results = {}
all_trades = {}

for name, func in strategies.items():
    trades_df = func()
    metrics = compute_metrics(trades_df)
    regime = regime_analysis(trades_df)
    perm_p = permutation_test(trades_df, metrics['sharpe'])
    gates, n_passed, verdict = validate_gates(metrics, regime, perm_p)

    result = {
        'sharpe': metrics['sharpe'],
        'sortino': metrics['sortino'],
        'total_return': metrics['total_return'],
        'max_drawdown': metrics['max_drawdown'],
        'n_trades': metrics['n_trades'],
        'profit_factor': metrics['profit_factor'],
        'win_rate': metrics['win_rate'],
        'cagr': metrics['cagr'],
        'final_equity': metrics['final_equity'],
        'regime_analysis': regime,
        'permutation_pvalue': perm_p,
        'gates': gates,
        'gates_passed': n_passed,
        'verdict': verdict
    }
    all_results[name] = result
    all_trades[name] = trades_df

    print(f"  {name}: Sharpe={metrics['sharpe']:.3f}, Return={metrics['total_return']*100:.1f}%, "
          f"WR={metrics['win_rate']*100:.1f}%, N={metrics['n_trades']}, DD={metrics['max_drawdown']*100:.1f}%, "
          f"Perm-p={perm_p:.3f}, Regime-gap={regime['regime_gap']:.3f} => {verdict}")

# Find best variant for adversarial reference
best_name = max(all_results, key=lambda k: all_results[k]['sharpe'])
print(f"\n  Best variant for adversarial reference: {best_name}")

# Strategy F: Random adversarial
trades_f = strategy_random(all_trades[best_name])
metrics_f = compute_metrics(trades_f)
regime_f = regime_analysis(trades_f)
perm_f = permutation_test(trades_f, metrics_f['sharpe'])
gates_f, n_passed_f, verdict_f = validate_gates(metrics_f, regime_f, perm_f)

result_f = {
    'sharpe': metrics_f['sharpe'],
    'sortino': metrics_f['sortino'],
    'total_return': metrics_f['total_return'],
    'max_drawdown': metrics_f['max_drawdown'],
    'n_trades': metrics_f['n_trades'],
    'profit_factor': metrics_f['profit_factor'],
    'win_rate': metrics_f['win_rate'],
    'cagr': metrics_f['cagr'],
    'final_equity': metrics_f['final_equity'],
    'regime_analysis': regime_f,
    'permutation_pvalue': perm_f,
    'gates': gates_f,
    'gates_passed': n_passed_f,
    'verdict': verdict_f,
    'reference_strategy': best_name,
    'note': 'Adversarial: random entry dates with same avg hold period and trade count as best variant'
}
all_results['F_RandomCalendar'] = result_f

print(f"  F_RandomCalendar: Sharpe={metrics_f['sharpe']:.3f}, Return={metrics_f['total_return']*100:.1f}%, "
      f"WR={metrics_f['win_rate']*100:.1f}%, N={metrics_f['n_trades']}, DD={metrics_f['max_drawdown']*100:.1f}%, "
      f"Perm-p={perm_f:.3f}, Regime-gap={regime_f['regime_gap']:.3f} => {verdict_f}")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Variant':<22} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>7} {'WR':>6} {'PF':>6} {'N':>5} {'Perm-p':>7} {'RGap':>6} {'Gates':>6} {'Verdict':<12}")
print("-" * 110)

for name, r in all_results.items():
    print(f"{name:<22} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['total_return']*100:>7.1f}% {r['max_drawdown']*100:>6.1f}% "
          f"{r['win_rate']*100:>5.1f}% {r['profit_factor']:>5.2f} {r['n_trades']:>5} {r['permutation_pvalue']:>7.3f} "
          f"{r['regime_analysis']['regime_gap']:>5.3f} {r['gates_passed']:>3}/5  {r['verdict']:<12}")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
output_path = '/home/jupiter/Lvl3Quant/data/calendar_anomaly_results.json'

# Convert for JSON serialization
json_results = {}
for name, r in all_results.items():
    jr = {k: v for k, v in r.items() if k not in ('daily_returns', 'trades_df')}
    # Convert gate keys
    if 'gates' in jr:
        jr['gates'] = {k: bool(v) for k, v in jr['gates'].items()}
    json_results[name] = jr

json_results['_metadata'] = {
    'run_date': datetime.now().isoformat(),
    'period': f'{START} to {END}',
    'starting_capital': CAPITAL,
    'slippage_pct': SLIPPAGE_PCT,
    'commission': COMMISSION,
    'n_permutations': N_PERMUTATIONS,
    'validation_gates': {
        'sharpe_threshold': SHARPE_GATE,
        'perm_pvalue_threshold': PERM_PVALUE_GATE,
        'regime_gap_threshold': REGIME_GAP_GATE,
        'max_drawdown_threshold': MAX_DD_GATE,
        'min_trades': MIN_TRADES_GATE
    }
}

with open(output_path, 'w') as f:
    json.dump(json_results, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done.")
