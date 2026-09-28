#!/usr/bin/env python3
"""
Seasonal Quality Patterns Backtest
Tests whether quality stocks exhibit exploitable seasonal patterns.
OOT: Jan 2022 – Jul 2026 | Starting Capital: $645
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
QUALITY_UNIVERSE = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
START_DATE = '2021-10-01'  # extra buffer for 200-SMA warmup
END_DATE = '2026-07-31'
OOT_START = '2022-01-01'
STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 0.02% each way
N_PERMUTATIONS = 1000
RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/seasonal_quality_patterns_results.json'

# ── Download Data ───────────────────────────────────────────────────────
print("Downloading price data...")
tickers_to_download = QUALITY_UNIVERSE + ['SPY']
data = {}
for ticker in tickers_to_download:
    try:
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 100:
            data[ticker] = df
            print(f"  {ticker}: {len(df)} days")
        else:
            print(f"  {ticker}: insufficient data ({len(df)} days), skipping")
    except Exception as e:
        print(f"  {ticker}: download failed ({e})")

spy = data.get('SPY')
if spy is None:
    raise RuntimeError("SPY download failed")

# Compute SPY 200-SMA for regime analysis
spy['SMA200'] = spy['Close'].rolling(200).mean()
spy['Bull'] = spy['Close'] > spy['SMA200']

print(f"\nLoaded {len(data)-1} quality stocks + SPY")

# ── Helper Functions ────────────────────────────────────────────────────

def compute_metrics(returns_series):
    """Compute Sharpe, Sortino, WR, PF, MDD, Total Return from a series of trade returns."""
    if len(returns_series) == 0:
        return {'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
                'max_drawdown': 0, 'total_return': 0, 'n_trades': 0}

    returns = np.array(returns_series)
    n_trades = len(returns)
    winners = returns[returns > 0]
    losers = returns[returns < 0]

    win_rate = len(winners) / n_trades if n_trades > 0 else 0
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (10.0 if gross_profit > 0 else 0)

    # Annualize: assume ~20 trades/year average for Sharpe calc
    mean_ret = returns.mean()
    std_ret = returns.std() if len(returns) > 1 else 1e-9
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9

    # Annualization factor based on avg trades per year
    years = 4.5  # Jan 2022 - Jul 2026
    trades_per_year = n_trades / years if years > 0 else n_trades
    ann_factor = np.sqrt(max(trades_per_year, 1))

    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 1e-9 else 0
    sortino = (mean_ret / downside_std) * ann_factor if downside_std > 1e-9 else 0

    # Max drawdown from cumulative equity
    cum = np.cumsum(returns)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = dd.min() if len(dd) > 0 else 0

    total_return = cum[-1] if len(cum) > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 3),
        'max_drawdown': round(max_dd, 4),
        'total_return': round(total_return, 4),
        'n_trades': n_trades
    }


def regime_analysis(trade_records, spy_df):
    """Split trades into bull/bear regime and compute Sharpe for each."""
    bull_rets = []
    bear_rets = []
    for rec in trade_records:
        entry_date = rec['entry_date']
        # Find closest SPY date
        idx = spy_df.index.get_indexer([pd.Timestamp(entry_date)], method='ffill')[0]
        if idx >= 0 and idx < len(spy_df):
            is_bull = spy_df['Bull'].iloc[idx]
            if is_bull:
                bull_rets.append(rec['return_pct'])
            else:
                bear_rets.append(rec['return_pct'])

    bull_metrics = compute_metrics(bull_rets) if bull_rets else {'sharpe': 0, 'n_trades': 0}
    bear_metrics = compute_metrics(bear_rets) if bear_rets else {'sharpe': 0, 'n_trades': 0}

    gap = abs(bull_metrics['sharpe'] - bear_metrics['sharpe'])
    max_s = max(abs(bull_metrics['sharpe']), abs(bear_metrics['sharpe']), 1e-9)
    regime_gap_ratio = gap / max_s

    return {
        'bull_sharpe': bull_metrics['sharpe'],
        'bull_trades': bull_metrics['n_trades'],
        'bear_sharpe': bear_metrics['sharpe'],
        'bear_trades': bear_metrics['n_trades'],
        'regime_gap': round(regime_gap_ratio, 3)
    }


def permutation_test(actual_return, trade_records, all_prices, n_perms=1000):
    """Shuffle entry dates and recompute total return to get p-value."""
    if len(trade_records) < 5:
        return 1.0

    # Get all valid trading dates from the OOT period
    valid_dates = all_prices.index[all_prices.index >= OOT_START].tolist()
    if len(valid_dates) < 20:
        return 1.0

    hold_days = []
    for rec in trade_records:
        hold_days.append(rec.get('hold_days', 5))

    perm_returns = []
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        perm_ret = 0
        n_sample = min(len(trade_records), len(valid_dates) // 2)
        random_indices = rng.choice(len(valid_dates) - max(hold_days) - 1, size=n_sample, replace=True)

        for j, idx in enumerate(random_indices):
            hold = hold_days[j % len(hold_days)]
            entry_idx = idx
            exit_idx = min(idx + hold, len(valid_dates) - 1)

            # Pick a random stock
            stock = rng.choice(QUALITY_UNIVERSE)
            if stock not in data:
                continue
            stock_df = data[stock]

            entry_date = valid_dates[entry_idx]
            exit_date = valid_dates[exit_idx]

            if entry_date in stock_df.index and exit_date in stock_df.index:
                p_entry = stock_df.loc[entry_date, 'Close']
                p_exit = stock_df.loc[exit_date, 'Close']
                ret = (p_exit / p_entry - 1) - 2 * SLIPPAGE_PCT
                perm_ret += ret

        perm_returns.append(perm_ret)

    perm_returns = np.array(perm_returns)
    p_value = (perm_returns >= actual_return).mean()
    return round(float(p_value), 4)


def execute_trades(signals, capital=STARTING_CAPITAL):
    """
    Given a list of signal dicts with {ticker, entry_date, exit_date},
    execute trades respecting position limits and capital constraints.
    Returns list of trade records with returns.
    """
    trades = []
    # Sort signals by entry date
    signals = sorted(signals, key=lambda x: x['entry_date'])

    equity = capital
    open_positions = []

    for sig in signals:
        ticker = sig['ticker']
        entry_date = pd.Timestamp(sig['entry_date'])
        exit_date = pd.Timestamp(sig['exit_date'])

        if ticker not in data:
            continue

        stock_df = data[ticker]

        # Close expired positions
        open_positions = [p for p in open_positions if p['exit_date'] > entry_date]

        # Check concurrent limit
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        # Find actual trading dates
        valid_entry = stock_df.index[stock_df.index >= entry_date]
        valid_exit = stock_df.index[stock_df.index >= exit_date]

        if len(valid_entry) == 0 or len(valid_exit) == 0:
            continue

        actual_entry = valid_entry[0]
        actual_exit = valid_exit[0]

        if actual_entry >= actual_exit:
            continue
        if actual_entry not in stock_df.index or actual_exit not in stock_df.index:
            continue

        entry_price = float(stock_df.loc[actual_entry, 'Close'])
        exit_price = float(stock_df.loc[actual_exit, 'Close'])

        # Position sizing
        trade_size = min(MAX_PER_TRADE, equity * 0.33)
        if trade_size < 10:
            continue

        shares = trade_size / entry_price

        # Apply slippage
        entry_cost = entry_price * (1 + SLIPPAGE_PCT)
        exit_proceeds = exit_price * (1 - SLIPPAGE_PCT)

        ret_pct = (exit_proceeds / entry_cost) - 1
        pnl = shares * entry_price * ret_pct

        equity += pnl
        hold_days = (actual_exit - actual_entry).days

        trade_rec = {
            'ticker': ticker,
            'entry_date': str(actual_entry.date()),
            'exit_date': str(actual_exit.date()),
            'entry_price': round(entry_price, 2),
            'exit_price': round(exit_price, 2),
            'return_pct': round(ret_pct, 6),
            'pnl': round(pnl, 2),
            'hold_days': hold_days
        }
        trades.append(trade_rec)
        open_positions.append({'exit_date': actual_exit})

    return trades


def get_trading_days(year, month=None):
    """Get trading days for a given year/month from SPY data."""
    mask = spy.index.year == year
    if month is not None:
        mask &= spy.index.month == month
    return spy.index[mask].tolist()


def get_nth_last_trading_day(year, month, n):
    """Get the nth-to-last trading day of month."""
    days = get_trading_days(year, month)
    if len(days) >= n:
        return days[-n]
    return days[0] if days else None


def get_nth_trading_day(year, month, n):
    """Get the nth trading day of month (1-indexed)."""
    days = get_trading_days(year, month)
    if len(days) >= n:
        return days[n-1]
    return days[-1] if days else None


def get_third_friday(year, month):
    """Get the 3rd Friday of a month."""
    import calendar
    c = calendar.Calendar()
    fridays = [d for d in c.itermonthdays2(year, month) if d[0] != 0 and d[1] == 4]
    if len(fridays) >= 3:
        return datetime(year, month, fridays[2][0])
    return None


# ── Strategy Signal Generators ──────────────────────────────────────────

def strategy_A_january_effect():
    """Buy first trading day of January, sell Jan 31."""
    signals = []
    for year in range(2022, 2027):
        jan_days = get_trading_days(year, 1)
        if len(jan_days) < 2:
            continue
        entry = jan_days[0]
        exit_d = jan_days[-1]
        for ticker in QUALITY_UNIVERSE:
            signals.append({'ticker': ticker, 'entry_date': entry, 'exit_date': exit_d})
    return signals


def strategy_B_sell_in_may():
    """Buy Nov 1, sell Apr 30 (good half)."""
    signals = []
    # Nov 2021 -> Apr 2022, Nov 2022 -> Apr 2023, ..., Nov 2025 -> Apr 2026
    for start_year in range(2021, 2026):
        nov_days = get_trading_days(start_year, 11)
        apr_days = get_trading_days(start_year + 1, 4)
        if not nov_days or not apr_days:
            continue
        entry = nov_days[0]
        exit_d = apr_days[-1]
        for ticker in QUALITY_UNIVERSE:
            signals.append({'ticker': ticker, 'entry_date': entry, 'exit_date': exit_d})
    return signals


def strategy_C_month_end_rebalancing():
    """Buy 3rd-to-last trading day of month, sell 3rd trading day of next month."""
    signals = []
    for year in range(2022, 2027):
        for month in range(1, 13):
            if year == 2026 and month > 7:
                break
            entry = get_nth_last_trading_day(year, month, 3)
            if entry is None:
                continue
            # Next month
            next_month = month + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year = year + 1
            exit_d = get_nth_trading_day(next_year, next_month, 3)
            if exit_d is None:
                continue
            for ticker in QUALITY_UNIVERSE:
                signals.append({'ticker': ticker, 'entry_date': entry, 'exit_date': exit_d})
    return signals


def strategy_D_santa_rally():
    """Buy Dec 15, sell Jan 5."""
    signals = []
    for year in range(2021, 2026):
        dec_days = get_trading_days(year, 12)
        jan_days = get_trading_days(year + 1, 1)
        if not dec_days or not jan_days:
            continue
        # Find closest to Dec 15
        target_entry = pd.Timestamp(f'{year}-12-15')
        entry_candidates = [d for d in dec_days if d >= target_entry]
        if not entry_candidates:
            continue
        entry = entry_candidates[0]
        # Find closest to Jan 5
        target_exit = pd.Timestamp(f'{year+1}-01-05')
        exit_candidates = [d for d in jan_days if d >= target_exit]
        if not exit_candidates:
            exit_d = jan_days[-1]
        else:
            exit_d = exit_candidates[0]
        for ticker in QUALITY_UNIVERSE:
            signals.append({'ticker': ticker, 'entry_date': entry, 'exit_date': exit_d})
    return signals


def strategy_E_tax_loss_recovery():
    """Buy Jan 15, hold 20 trading days."""
    signals = []
    for year in range(2022, 2027):
        jan_days = get_trading_days(year, 1)
        if len(jan_days) < 5:
            continue
        target = pd.Timestamp(f'{year}-01-15')
        entry_candidates = [d for d in jan_days if d >= target]
        if not entry_candidates:
            continue
        entry = entry_candidates[0]
        # 20 trading days forward
        all_days = spy.index[spy.index >= entry].tolist()
        if len(all_days) < 21:
            continue
        exit_d = all_days[20]
        for ticker in QUALITY_UNIVERSE:
            signals.append({'ticker': ticker, 'entry_date': entry, 'exit_date': exit_d})
    return signals


def strategy_F_quad_witching():
    """Buy Monday after quad witching (3rd Friday of Mar/Jun/Sep/Dec), hold 5 days."""
    signals = []
    witching_months = [3, 6, 9, 12]
    for year in range(2022, 2027):
        for month in witching_months:
            if year == 2026 and month > 7:
                break
            third_fri = get_third_friday(year, month)
            if third_fri is None:
                continue
            # Monday after = next trading day after Friday
            target_monday = pd.Timestamp(third_fri) + timedelta(days=3)
            all_days = spy.index[spy.index >= target_monday].tolist()
            if len(all_days) < 6:
                continue
            entry = all_days[0]
            exit_d = all_days[5]
            for ticker in QUALITY_UNIVERSE:
                signals.append({'ticker': ticker, 'entry_date': entry, 'exit_date': exit_d})
    return signals


def strategy_G_monday_buy():
    """Buy Monday close, sell Friday close. Every week."""
    signals = []
    # Get all Mondays in OOT period
    oot_days = spy.index[spy.index >= OOT_START].tolist()
    for d in oot_days:
        if d.dayofweek == 0:  # Monday
            # Find the Friday of same week
            week_days = [x for x in oot_days if x >= d and x <= d + timedelta(days=5) and x.dayofweek == 4]
            if not week_days:
                continue
            friday = week_days[0]
            # Rotate through stocks to manage positions
            ticker_idx = (d.month * 31 + d.day) % len(QUALITY_UNIVERSE)
            # Pick 3 stocks per week (max concurrent = 3)
            for i in range(3):
                idx = (ticker_idx + i) % len(QUALITY_UNIVERSE)
                signals.append({
                    'ticker': QUALITY_UNIVERSE[idx],
                    'entry_date': d,
                    'exit_date': friday
                })
    return signals


def strategy_H_summer_dip():
    """Buy quality stocks on any 3%+ dip in July-August."""
    signals = []
    for ticker in QUALITY_UNIVERSE:
        if ticker not in data:
            continue
        stock_df = data[ticker]
        for year in range(2022, 2027):
            for month in [7, 8]:
                if year == 2026 and month > 7:
                    break
                month_days = stock_df.index[(stock_df.index.year == year) & (stock_df.index.month == month)]
                for d in month_days:
                    idx_pos = stock_df.index.get_loc(d)
                    if idx_pos < 5:
                        continue
                    # Check for 3% dip from recent 5-day high
                    recent = stock_df['Close'].iloc[idx_pos-5:idx_pos]
                    current = stock_df['Close'].iloc[idx_pos]
                    recent_high = recent.max()
                    dip = (current / recent_high) - 1
                    if dip <= -0.03:
                        # Hold 10 trading days for recovery
                        future_days = stock_df.index[stock_df.index > d].tolist()
                        if len(future_days) < 10:
                            continue
                        exit_d = future_days[9]
                        signals.append({
                            'ticker': ticker,
                            'entry_date': d,
                            'exit_date': exit_d
                        })
    return signals


# ── Run All Strategies ──────────────────────────────────────────────────

strategies = {
    'A': ('January Effect', strategy_A_january_effect),
    'B': ('Sell in May (Buy Nov-Apr)', strategy_B_sell_in_may),
    'C': ('Month-End Rebalancing', strategy_C_month_end_rebalancing),
    'D': ('Santa Rally', strategy_D_santa_rally),
    'E': ('Tax Loss Recovery', strategy_E_tax_loss_recovery),
    'F': ('Quadruple Witching Bounce', strategy_F_quad_witching),
    'G': ('Day-of-Week (Monday Buy)', strategy_G_monday_buy),
    'H': ('Summer Doldrums Fade', strategy_H_summer_dip),
}

results = {}

print("\n" + "="*90)
print("SEASONAL QUALITY PATTERNS BACKTEST")
print(f"OOT Period: Jan 2022 - Jul 2026 | Capital: ${STARTING_CAPITAL} | Max/Trade: ${MAX_PER_TRADE}")
print("="*90)

for key, (name, gen_func) in strategies.items():
    print(f"\n--- Variant {key}: {name} ---")

    # Generate signals
    signals = gen_func()
    print(f"  Raw signals: {len(signals)}")

    # Execute trades
    trades = execute_trades(signals)
    print(f"  Executed trades: {len(trades)}")

    if len(trades) == 0:
        results[key] = {
            'name': name,
            'metrics': compute_metrics([]),
            'regime': {'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 0, 'bull_trades': 0, 'bear_trades': 0},
            'perm_p_value': 1.0,
            'gates_passed': 0,
            'gate_details': {}
        }
        continue

    # Compute metrics
    trade_returns = [t['return_pct'] for t in trades]
    metrics = compute_metrics(trade_returns)
    print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | "
          f"WR: {metrics['win_rate']:.1%} | PF: {metrics['profit_factor']:.2f} | "
          f"MDD: {metrics['max_drawdown']:.2%} | Return: {metrics['total_return']:.2%}")

    # Regime analysis
    regime = regime_analysis(trades, spy)
    print(f"  Bull Sharpe: {regime['bull_sharpe']:.3f} ({regime['bull_trades']} trades) | "
          f"Bear Sharpe: {regime['bear_sharpe']:.3f} ({regime['bear_trades']} trades) | "
          f"Gap: {regime['regime_gap']:.3f}")

    # Permutation test (for timing-based strategies)
    total_return = sum(trade_returns)
    perm_p = permutation_test(total_return, trades, spy, N_PERMUTATIONS)
    print(f"  Permutation p-value: {perm_p:.4f}")

    # 5-Gate Validation
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime['regime_gap'] < 0.5,
        'mdd_gt_neg50pct': metrics['max_drawdown'] > -0.50,
        'trades_gte_20': metrics['n_trades'] >= 20
    }
    gates_passed = sum(gates.values())
    print(f"  Gates: {gates_passed}/5 — {gates}")

    results[key] = {
        'name': name,
        'metrics': metrics,
        'regime': regime,
        'perm_p_value': perm_p,
        'gates_passed': gates_passed,
        'gate_details': {k: bool(v) for k, v in gates.items()}
    }

# ── Summary Table ───────────────────────────────────────────────────────

print("\n\n" + "="*120)
print(f"{'Var':<4} {'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'WR':>7} {'PF':>7} {'MDD':>8} {'Return':>8} {'Trades':>7} {'Perm-p':>7} {'RGap':>6} {'Gates':>6}")
print("-"*120)

for key in sorted(results.keys()):
    r = results[key]
    m = r['metrics']
    rg = r['regime']
    marker = " ***" if r['gates_passed'] >= 4 else ""
    print(f"  {key}  {r['name']:<30} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
          f"{m['win_rate']:>6.1%} {m['profit_factor']:>7.2f} {m['max_drawdown']:>7.2%} "
          f"{m['total_return']:>7.2%} {m['n_trades']:>7d} {r['perm_p_value']:>7.4f} "
          f"{rg['regime_gap']:>6.3f} {r['gates_passed']:>3d}/5{marker}")

print("-"*120)

# Best strategy
best = max(results.items(), key=lambda x: x[1]['gates_passed'] * 100 + x[1]['metrics']['sharpe'])
print(f"\nBest: Variant {best[0]} ({best[1]['name']}) — "
      f"{best[1]['gates_passed']}/5 gates, Sharpe {best[1]['metrics']['sharpe']:.3f}")

# ── Save Results ────────────────────────────────────────────────────────

output = {
    'backtest': 'seasonal_quality_patterns',
    'oot_period': 'Jan 2022 - Jul 2026',
    'starting_capital': STARTING_CAPITAL,
    'universe': QUALITY_UNIVERSE,
    'run_timestamp': datetime.now().isoformat(),
    'variants': results
}

Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_PATH}")
