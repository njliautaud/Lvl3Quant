#!/usr/bin/env python3
"""
Drawdown Recovery Backtest v2
=============================
6 variants testing buying during market drawdowns with recovery signals.

A. Drawdown Depth Buy: QQQ 10%+ below 52w high, first up day after 3+ down days
B. Breadth Thrust: RSP/SPY ratio proxy for breadth — drops 3%+ then reverses
C. VIX Mean Reversion: VIX spikes 50%+ above 20d MA then closes back below
D. Volume Capitulation: QQQ drops 3%+ on 2x volume, next day flat/up
E. Recovery Velocity: QQQ drops 7%+ from 20d high, rallies 2%+ in 2 days → buy TQQQ
F. Multi-Signal Recovery: 3 of 5 oversold conditions met → buy QQQ
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
DATA_START = '2020-06-01'  # extra history for 52w high, SMAs
SLIPPAGE = 0.0002  # 0.02% per trade (each way)
N_PERMUTATIONS = 1000

GATES = {
    'sharpe_min': 0.5,
    'perm_p_max': 0.05,
    'regime_gap_max': 0.5,
    'max_dd_floor': -0.50,  # max DD must be better than -50%
    'min_trades': 20,
}


# ─── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download all needed tickers."""
    tickers = ['QQQ', 'TQQQ', 'SPY', 'RSP', '^VIX']
    print("Downloading data...")
    data = {}
    for t in tickers:
        df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[t.replace('^', '')] = df
    print(f"  Downloaded {len(data)} tickers")
    for k, v in data.items():
        print(f"    {k}: {len(v)} rows, {v.index[0].date()} to {v.index[-1].date()}")
    return data


# ─── Helper Functions ────────────────────────────────────────────────────────
def compute_rsi(prices, period=14):
    """Compute RSI."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def bollinger_bands(prices, period=20, num_std=2):
    """Compute Bollinger Bands, return lower band."""
    sma = prices.rolling(period).mean()
    std = prices.rolling(period).std()
    lower = sma - num_std * std
    return lower


def run_backtest(signal_series, prices, hold_days, label):
    """
    Run a backtest given a boolean signal series.
    signal_series: True on signal day T (buy happens T+1 open → hold for hold_days).
    prices: Close prices of the instrument being traded.
    """
    # Shift signal by 1 day (signal on T → trade on T+1)
    trade_signal = signal_series.shift(1).fillna(False).astype(bool)

    # Align to OOT period
    oot_mask = (prices.index >= OOT_START) & (prices.index <= OOT_END)
    prices_oot = prices[oot_mask].copy()
    trade_signal_oot = trade_signal.reindex(prices_oot.index, fill_value=False)

    daily_returns = prices_oot.pct_change().fillna(0)

    # Track positions
    trades = []
    position_mask = pd.Series(False, index=prices_oot.index)
    days_in_trade = 0
    in_trade = False
    entry_idx = None
    entry_price = None

    for i, date in enumerate(prices_oot.index):
        if in_trade:
            days_in_trade += 1
            if days_in_trade >= hold_days:
                # Exit
                exit_price = prices_oot.iloc[i] * (1 - SLIPPAGE)
                trade_ret = (exit_price / entry_price) - 1
                trades.append({
                    'entry_date': entry_idx,
                    'exit_date': date,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'return': trade_ret,
                    'hold_days': days_in_trade,
                })
                in_trade = False
                days_in_trade = 0
            else:
                position_mask.iloc[i] = True
        elif trade_signal_oot.iloc[i] and not in_trade:
            # Enter
            entry_price = prices_oot.iloc[i] * (1 + SLIPPAGE)
            entry_idx = date
            in_trade = True
            days_in_trade = 0
            position_mask.iloc[i] = True

    # Close any open trade at end
    if in_trade:
        exit_price = prices_oot.iloc[-1] * (1 - SLIPPAGE)
        trade_ret = (exit_price / entry_price) - 1
        trades.append({
            'entry_date': entry_idx,
            'exit_date': prices_oot.index[-1],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': trade_ret,
            'hold_days': days_in_trade,
        })

    if len(trades) == 0:
        print(f"  [{label}] WARNING: 0 trades generated!")
        return None

    trades_df = pd.DataFrame(trades)

    # Build equity curve
    strategy_daily_ret = pd.Series(0.0, index=prices_oot.index)
    for _, t in trades_df.iterrows():
        mask = (prices_oot.index >= t['entry_date']) & (prices_oot.index <= t['exit_date'])
        trade_days = prices_oot.index[mask]
        for j, d in enumerate(trade_days):
            if j > 0:
                strategy_daily_ret.loc[d] = daily_returns.loc[d]

    equity = INITIAL_CAPITAL * (1 + strategy_daily_ret).cumprod()

    return {
        'trades_df': trades_df,
        'equity': equity,
        'daily_returns': strategy_daily_ret,
        'position_mask': position_mask,
    }


def compute_metrics(result, benchmark_returns, spy_sma_bull, label):
    """Compute all required metrics from backtest result."""
    if result is None:
        return None

    trades_df = result['trades_df']
    equity = result['equity']
    daily_ret = result['daily_returns']

    n_trades = len(trades_df)
    wins = trades_df[trades_df['return'] > 0]
    losses = trades_df[trades_df['return'] <= 0]

    total_return = (equity.iloc[-1] / INITIAL_CAPITAL) - 1
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Annual vol & Sharpe
    annual_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-9
    sortino = (daily_ret.mean() * 252) / downside_std

    # Max drawdown
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if abs(max_dd) > 0 else 0

    # Win rate, profit factor
    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = wins['return'].mean() if len(wins) > 0 else 0
    avg_loss = abs(losses['return'].mean()) if len(losses) > 0 else 1e-9
    gross_profit = wins['return'].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses['return'].sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # QQQ correlation
    aligned_bench = benchmark_returns.reindex(daily_ret.index).fillna(0)
    correlation = daily_ret.corr(aligned_bench) if daily_ret.std() > 0 else 0

    # Bear 2022 performance (2022-01-01 to 2022-12-31)
    bear_mask = (daily_ret.index >= '2022-01-01') & (daily_ret.index <= '2022-12-31')
    bear_ret = daily_ret[bear_mask]
    bear_total = (1 + bear_ret).prod() - 1

    # Regime-stratified Sharpe
    bull_mask = spy_sma_bull.reindex(daily_ret.index).fillna(False)
    bear_regime_mask = ~bull_mask

    bull_ret = daily_ret[bull_mask]
    bear_regime_ret = daily_ret[bear_regime_mask]

    sharpe_bull = (bull_ret.mean() * 252) / (bull_ret.std() * np.sqrt(252)) if bull_ret.std() > 0 and len(bull_ret) > 20 else 0
    sharpe_bear = (bear_regime_ret.mean() * 252) / (bear_regime_ret.std() * np.sqrt(252)) if bear_regime_ret.std() > 0 and len(bear_regime_ret) > 20 else 0

    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)

    return {
        'label': label,
        'total_return_pct': round(total_return * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'annual_vol_pct': round(annual_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'n_trades': n_trades,
        'win_rate_pct': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 3),
        'avg_win_pct': round(avg_win * 100, 2),
        'avg_loss_pct': round(avg_loss * 100, 2),
        'qqq_correlation': round(correlation, 3),
        'bear_2022_return_pct': round(bear_total * 100, 2),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 3),
    }


def permutation_test(result, prices_oot, hold_days, n_iter=N_PERMUTATIONS):
    """Permutation test: randomize entry dates, compare Sharpe."""
    if result is None:
        return 1.0

    actual_sharpe = result['daily_returns'].mean() / result['daily_returns'].std() if result['daily_returns'].std() > 0 else 0
    n_trades = len(result['trades_df'])

    if n_trades == 0:
        return 1.0

    valid_start = prices_oot.index[0]
    valid_end = prices_oot.index[-hold_days - 1] if hold_days < len(prices_oot) else prices_oot.index[-2]
    valid_dates = prices_oot.index[(prices_oot.index >= valid_start) & (prices_oot.index <= valid_end)]

    if len(valid_dates) < n_trades:
        return 1.0

    count_better = 0
    daily_returns_all = prices_oot.pct_change().fillna(0)

    for _ in range(n_iter):
        # Random entry dates
        random_entries = np.random.choice(len(valid_dates), size=n_trades, replace=False)
        random_entries.sort()

        rand_daily = pd.Series(0.0, index=prices_oot.index)
        for idx in random_entries:
            entry_date = valid_dates[idx]
            entry_pos = prices_oot.index.get_loc(entry_date)
            exit_pos = min(entry_pos + hold_days, len(prices_oot) - 1)
            for j in range(entry_pos + 1, exit_pos + 1):
                rand_daily.iloc[j] = daily_returns_all.iloc[j]

        rand_sharpe = rand_daily.mean() / rand_daily.std() if rand_daily.std() > 0 else 0
        if rand_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_iter


def validate_gates(metrics, perm_p):
    """Check all validation gates."""
    if metrics is None:
        return {'pass': False, 'failures': ['No trades']}

    failures = []
    if metrics['sharpe'] < GATES['sharpe_min']:
        failures.append(f"Sharpe {metrics['sharpe']:.3f} < {GATES['sharpe_min']}")
    if perm_p >= GATES['perm_p_max']:
        failures.append(f"Perm p={perm_p:.3f} >= {GATES['perm_p_max']}")
    if metrics['regime_gap'] >= GATES['regime_gap_max']:
        failures.append(f"Regime gap {metrics['regime_gap']:.3f} >= {GATES['regime_gap_max']}")
    if metrics['max_drawdown_pct'] < GATES['max_dd_floor'] * 100:
        failures.append(f"Max DD {metrics['max_drawdown_pct']:.1f}% worse than {GATES['max_dd_floor']*100}%")
    if metrics['n_trades'] < GATES['min_trades']:
        failures.append(f"Trades {metrics['n_trades']} < {GATES['min_trades']}")

    return {'pass': len(failures) == 0, 'failures': failures}


# ─── Strategy Signal Generators ─────────────────────────────────────────────

def strategy_A_drawdown_depth(data):
    """
    A) Drawdown Depth Buy:
    QQQ 10%+ below 52-week high AND first up-day after 3+ consecutive down days.
    Buy QQQ, hold 20 days.
    """
    qqq = data['QQQ']['Close'].copy()
    high_52w = qqq.rolling(252, min_periods=50).max()
    dd_from_high = (qqq - high_52w) / high_52w

    # Drawdown >= 10%
    deep_dd = dd_from_high <= -0.10

    # Consecutive down days
    daily_change = qqq.diff()
    down_day = daily_change < 0
    up_day = daily_change > 0

    # Count consecutive down days
    consec_down = pd.Series(0, index=qqq.index)
    for i in range(1, len(consec_down)):
        if down_day.iloc[i]:
            consec_down.iloc[i] = consec_down.iloc[i-1] + 1
        else:
            consec_down.iloc[i] = 0

    # First up day after 3+ consecutive down days
    prev_consec_down = consec_down.shift(1).fillna(0)
    first_up_after_3down = up_day & (prev_consec_down >= 3)

    signal = deep_dd & first_up_after_3down
    print(f"  [A] Drawdown Depth: {signal.sum()} raw signals")
    return signal, data['QQQ']['Close'], 20


def strategy_B_breadth_thrust(data):
    """
    B) Breadth Thrust (RSP/SPY ratio proxy):
    RSP/SPY drops 3%+ from 20d high then reverses up within 5 days.
    Buy QQQ, hold 30 days.
    """
    rsp = data['RSP']['Close'].copy()
    spy = data['SPY']['Close'].copy()

    # Align indices
    common = rsp.index.intersection(spy.index)
    rsp = rsp.loc[common]
    spy = spy.loc[common]

    ratio = rsp / spy
    ratio_20d_high = ratio.rolling(20, min_periods=10).max()
    ratio_dd = (ratio - ratio_20d_high) / ratio_20d_high

    # Ratio dropped 3%+ from 20d high
    deep_drop = ratio_dd <= -0.03

    # Reversal: ratio was in deep_drop within last 5 days AND now ratio is rising
    was_deep_recent = deep_drop.rolling(5, min_periods=1).max().fillna(0).astype(bool)
    ratio_rising = ratio.diff() > 0
    ratio_not_deep_now = ratio_dd > -0.03  # recovered somewhat

    signal = was_deep_recent & ratio_rising & ratio_not_deep_now

    # Reindex to QQQ
    signal = signal.reindex(data['QQQ']['Close'].index, fill_value=False)
    print(f"  [B] Breadth Thrust: {signal.sum()} raw signals")
    return signal, data['QQQ']['Close'], 30


def strategy_C_vix_mean_reversion(data):
    """
    C) VIX Mean Reversion:
    VIX spikes 50%+ above 20d MA then closes back below the MA.
    Buy QQQ, hold 15 days.
    """
    vix = data['VIX']['Close'].copy()
    qqq = data['QQQ']['Close'].copy()

    vix_ma20 = vix.rolling(20, min_periods=10).mean()
    vix_above_pct = (vix - vix_ma20) / vix_ma20

    # Was 50%+ above MA recently (within last 10 days)
    spiked = vix_above_pct >= 0.50
    was_spiked = spiked.rolling(10, min_periods=1).max().fillna(0).astype(bool)

    # Now below MA
    below_ma = vix < vix_ma20

    # Previously above MA (yesterday)
    was_above = (vix.shift(1) >= vix_ma20.shift(1))

    signal = was_spiked & below_ma & was_above

    # Reindex to QQQ
    signal = signal.reindex(qqq.index, fill_value=False)
    print(f"  [C] VIX Mean Reversion: {signal.sum()} raw signals")
    return signal, qqq, 15


def strategy_D_volume_capitulation(data):
    """
    D) Volume Capitulation:
    QQQ drops 3%+ on volume > 2x 20d avg, next day flat or up.
    Buy QQQ, hold 10 days.
    """
    qqq_close = data['QQQ']['Close'].copy()
    qqq_vol = data['QQQ']['Volume'].copy()

    daily_ret = qqq_close.pct_change()
    avg_vol_20 = qqq_vol.rolling(20, min_periods=10).mean()

    # Big drop day: down 3%+ on 2x volume
    big_drop = (daily_ret <= -0.03) & (qqq_vol > 2 * avg_vol_20)

    # Next day flat or up
    next_day_ret = daily_ret.shift(-1)
    next_day_ok = next_day_ret >= 0

    # Signal is on the NEXT day (the flat/up day) — that's when we confirm
    # So signal is: yesterday was big_drop AND today is flat/up
    signal = big_drop.shift(1).fillna(False) & (daily_ret >= 0)

    print(f"  [D] Volume Capitulation: {signal.sum()} raw signals")
    return signal, qqq_close, 10


def strategy_E_recovery_velocity(data):
    """
    E) Recovery Velocity:
    QQQ drops 7%+ from 20d high, then rallies 2%+ in 2 days from the low.
    Buy TQQQ (3x leverage), hold 5 days.
    """
    qqq = data['QQQ']['Close'].copy()
    tqqq = data['TQQQ']['Close'].copy()

    high_20d = qqq.rolling(20, min_periods=10).max()
    dd_from_20d = (qqq - high_20d) / high_20d

    # 7%+ drawdown
    deep_dd = dd_from_20d <= -0.07

    # 2-day rally from recent low
    low_2d = qqq.rolling(2, min_periods=1).min()
    rally_2d = (qqq - low_2d) / low_2d

    # Was in deep drawdown recently (within last 5 days) and now rallying 2%+
    was_deep = deep_dd.rolling(5, min_periods=1).max().fillna(0).astype(bool)
    rallying = rally_2d >= 0.02

    signal = was_deep & rallying

    # Reindex to TQQQ
    signal = signal.reindex(tqqq.index, fill_value=False)
    print(f"  [E] Recovery Velocity: {signal.sum()} raw signals")
    return signal, tqqq, 5


def strategy_F_multi_signal(data):
    """
    F) Multi-Signal Recovery:
    At least 3 of 5 conditions met:
    (1) QQQ 10%+ below 52w high
    (2) VIX > 25
    (3) RSI(14) < 30
    (4) QQQ below lower Bollinger Band (20,2)
    (5) Volume > 1.5x average on a down day
    Buy QQQ, hold 15 days.
    """
    qqq = data['QQQ']['Close'].copy()
    vix = data['VIX']['Close'].copy()
    qqq_vol = data['QQQ']['Volume'].copy()

    # Condition 1: 10%+ below 52w high
    high_52w = qqq.rolling(252, min_periods=50).max()
    dd_52w = (qqq - high_52w) / high_52w
    c1 = (dd_52w <= -0.10).astype(int)

    # Condition 2: VIX > 25
    vix_aligned = vix.reindex(qqq.index, method='ffill')
    c2 = (vix_aligned > 25).astype(int)

    # Condition 3: RSI(14) < 30
    rsi = compute_rsi(qqq, 14)
    c3 = (rsi < 30).astype(int)

    # Condition 4: Below lower Bollinger Band
    lower_bb = bollinger_bands(qqq, 20, 2)
    c4 = (qqq < lower_bb).astype(int)

    # Condition 5: Volume > 1.5x average on down day
    avg_vol_20 = qqq_vol.rolling(20, min_periods=10).mean()
    daily_ret = qqq.pct_change()
    c5 = ((qqq_vol > 1.5 * avg_vol_20) & (daily_ret < 0)).astype(int)

    score = c1 + c2 + c3 + c4 + c5
    signal = score >= 3

    print(f"  [F] Multi-Signal: {signal.sum()} raw signals")
    print(f"      Condition counts: C1={c1.sum()}, C2={c2.sum()}, C3={c3.sum()}, C4={c4.sum()}, C5={c5.sum()}")
    return signal, qqq, 15


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print("DRAWDOWN RECOVERY BACKTEST v2")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${INITIAL_CAPITAL}")
    print(f"Slippage: {SLIPPAGE*100:.2f}% per side")
    print("=" * 80)

    data = download_data()

    # Benchmark: QQQ buy & hold
    qqq = data['QQQ']['Close']
    qqq_oot = qqq[(qqq.index >= OOT_START) & (qqq.index <= OOT_END)]
    qqq_returns = qqq_oot.pct_change().fillna(0)
    qqq_equity = INITIAL_CAPITAL * (1 + qqq_returns).cumprod()
    qqq_total_ret = (qqq_equity.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    qqq_years = (qqq_oot.index[-1] - qqq_oot.index[0]).days / 365.25
    qqq_cagr = ((1 + qqq_total_ret/100) ** (1/qqq_years) - 1) * 100
    qqq_sharpe = (qqq_returns.mean() * 252) / (qqq_returns.std() * np.sqrt(252))

    print(f"\nBenchmark QQQ B&H: {qqq_total_ret:.1f}% total, {qqq_cagr:.1f}% CAGR, Sharpe {qqq_sharpe:.3f}")

    # SPY regime: bull = above 200 SMA
    spy = data['SPY']['Close']
    spy_sma200 = spy.rolling(200, min_periods=100).mean()
    spy_bull = spy > spy_sma200

    # Run all strategies
    strategies = [
        ('A_Drawdown_Depth', strategy_A_drawdown_depth),
        ('B_Breadth_Thrust', strategy_B_breadth_thrust),
        ('C_VIX_MeanReversion', strategy_C_vix_mean_reversion),
        ('D_Volume_Capitulation', strategy_D_volume_capitulation),
        ('E_Recovery_Velocity', strategy_E_recovery_velocity),
        ('F_Multi_Signal', strategy_F_multi_signal),
    ]

    all_results = {}
    summary_table = []

    for name, strategy_fn in strategies:
        print(f"\n{'─'*60}")
        print(f"Strategy: {name}")
        print(f"{'─'*60}")

        signal, trade_prices, hold_days = strategy_fn(data)
        result = run_backtest(signal, trade_prices, hold_days, name)
        metrics = compute_metrics(result, qqq_returns, spy_bull, name)

        if metrics is None:
            print(f"  SKIPPED: No trades generated")
            all_results[name] = {
                'metrics': None,
                'perm_p': 1.0,
                'gates': {'pass': False, 'failures': ['No trades']},
            }
            continue

        # Permutation test
        prices_oot = trade_prices[(trade_prices.index >= OOT_START) & (trade_prices.index <= OOT_END)]
        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
        perm_p = permutation_test(result, prices_oot, hold_days)

        # Validation gates
        gates = validate_gates(metrics, perm_p)

        all_results[name] = {
            'metrics': metrics,
            'perm_p': round(perm_p, 4),
            'gates': gates,
        }

        # Print results
        m = metrics
        status = "PASS" if gates['pass'] else "FAIL"
        print(f"\n  {'='*50}")
        print(f"  [{status}] {name}")
        print(f"  {'='*50}")
        print(f"  Total Return: {m['total_return_pct']:.1f}% | CAGR: {m['cagr_pct']:.1f}%")
        print(f"  Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f}")
        print(f"  Max DD: {m['max_drawdown_pct']:.1f}% | Calmar: {m['calmar']:.3f}")
        print(f"  Annual Vol: {m['annual_vol_pct']:.1f}%")
        print(f"  Trades: {m['n_trades']} | Win Rate: {m['win_rate_pct']:.1f}%")
        print(f"  Profit Factor: {m['profit_factor']:.3f}")
        print(f"  Avg Win: {m['avg_win_pct']:.2f}% | Avg Loss: {m['avg_loss_pct']:.2f}%")
        print(f"  QQQ Correlation: {m['qqq_correlation']:.3f}")
        print(f"  Bear 2022 Return: {m['bear_2022_return_pct']:.1f}%")
        print(f"  Sharpe Bull: {m['sharpe_bull']:.3f} | Sharpe Bear: {m['sharpe_bear']:.3f}")
        print(f"  Regime Gap: {m['regime_gap']:.3f}")
        print(f"  Permutation p: {perm_p:.4f}")
        if not gates['pass']:
            print(f"  Gate Failures: {', '.join(gates['failures'])}")

        summary_table.append({
            'strategy': name,
            'sharpe': m['sharpe'],
            'sortino': m['sortino'],
            'total_return': m['total_return_pct'],
            'max_dd': m['max_drawdown_pct'],
            'trades': m['n_trades'],
            'win_rate': m['win_rate_pct'],
            'perm_p': perm_p,
            'regime_gap': m['regime_gap'],
            'pass': status,
        })

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Strategy':<25} {'Sharpe':>7} {'Sortino':>8} {'Return%':>8} {'MaxDD%':>7} {'Trades':>6} {'WR%':>5} {'Perm-p':>7} {'RGap':>5} {'Gate':>5}")
    print("-" * 90)
    for row in summary_table:
        print(f"{row['strategy']:<25} {row['sharpe']:>7.3f} {row['sortino']:>8.3f} {row['total_return']:>8.1f} {row['max_dd']:>7.1f} {row['trades']:>6} {row['win_rate']:>5.1f} {row['perm_p']:>7.4f} {row['regime_gap']:>5.3f} {row['pass']:>5}")

    # Benchmark row
    print(f"{'QQQ_BuyHold':<25} {qqq_sharpe:>7.3f} {'--':>8} {qqq_total_ret:>8.1f} {'--':>7} {'--':>6} {'--':>5} {'--':>7} {'--':>5} {'--':>5}")

    # Passing strategies
    passing = [name for name, r in all_results.items() if r['gates']['pass']]
    print(f"\nPassing strategies: {passing if passing else 'NONE'}")

    # Save results
    output = {
        'metadata': {
            'backtest': 'drawdown_recovery_v2',
            'oot_start': OOT_START,
            'oot_end': OOT_END,
            'initial_capital': INITIAL_CAPITAL,
            'slippage_pct': SLIPPAGE * 100,
            'n_permutations': N_PERMUTATIONS,
            'run_date': datetime.now().isoformat(),
            'gates': GATES,
        },
        'benchmark': {
            'qqq_total_return_pct': round(qqq_total_ret, 2),
            'qqq_cagr_pct': round(qqq_cagr, 2),
            'qqq_sharpe': round(qqq_sharpe, 3),
        },
        'strategies': {},
    }

    for name, r in all_results.items():
        output['strategies'][name] = {
            'metrics': r['metrics'],
            'permutation_p': r['perm_p'],
            'gates_pass': r['gates']['pass'],
            'gate_failures': r['gates']['failures'] if not r['gates']['pass'] else [],
        }

    output_path = '/home/jupiter/Lvl3Quant/data/drawdown_recovery_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
