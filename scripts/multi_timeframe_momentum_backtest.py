#!/usr/bin/env python3
"""
Multi-Timeframe Momentum Backtest
==================================
Combines short, medium, and long-term momentum signals for better timing.
Walk-forward OOT: Jan 2022 - Jul 2026. 5-gate validation.

Variants:
  A) Classic 12-1 Momentum
  B) Triple Momentum Score (1m/3m/6m composite)
  C) Momentum + Mean Reversion Timing (6m mom + RSI dip entry)
  D) Acceleration Momentum (3m - 6m return acceleration)
  E) Momentum with Crash Stop (12-1 + crash protection via GLD)
  F) Adaptive Lookback Momentum (VIX-based lookback selection)
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────
CAPITAL = 645.0
NUM_POSITIONS = 3
SLIPPAGE_PCT = 0.02 / 100  # 0.02%
COMMISSION = 0.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-30'
DATA_START = '2020-06-01'  # need lookback before OOT
PERM_ITERATIONS = 1000
SMA_200_PERIOD = 200

GROWTH = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'CRM', 'PLTR', 'UBER', 'COIN']
SAFE_HAVENS = ['GLD', 'TLT', 'SHY']
INDEX = ['SPY', 'QQQ']
ALL_TICKERS = GROWTH + SAFE_HAVENS + INDEX

np.random.seed(42)


def download_data():
    """Download price data for all tickers."""
    print("Downloading price data...")
    data = yf.download(ALL_TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')
    # Forward fill small gaps, then drop any remaining
    close = close.ffill(limit=5)
    return close


def compute_returns(close, period):
    """Compute period returns."""
    return close.pct_change(period)


def compute_rsi(series, period=5):
    """Compute RSI for a series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def get_regime(spy_close, date):
    """Bull if SPY > 200-SMA, else Bear."""
    loc = spy_close.index.get_loc(date)
    if loc < SMA_200_PERIOD:
        return 'bull'
    sma = spy_close.iloc[loc - SMA_200_PERIOD + 1:loc + 1].mean()
    return 'bull' if spy_close.iloc[loc] > sma else 'bear'


def get_monthly_rebal_dates(close, start, end):
    """Get month-end rebalance dates within OOT period."""
    idx = close.loc[start:end].index
    # Group by year-month, take last trading day
    monthly = idx.to_series().groupby([idx.year, idx.month]).last()
    return monthly.values


def apply_slippage(price, direction='buy'):
    """Apply slippage to execution price."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


def run_backtest(close, selection_fn, variant_name):
    """
    Generic monthly rebalance backtest.

    selection_fn(close, date, current_holdings) -> list of (ticker, weight) tuples
        where weight sums to ~1.0 for fully invested, <1.0 if some cash.

    Returns list of trades and equity curve.
    """
    spy_close = close['SPY']
    rebal_dates = get_monthly_rebal_dates(close, OOT_START, OOT_END)

    equity = CAPITAL
    cash = CAPITAL
    holdings = {}  # ticker -> {'shares': n, 'entry_price': p, 'entry_date': d}
    trades = []
    equity_curve = []

    all_dates = close.loc[OOT_START:OOT_END].index

    rebal_set = set(pd.DatetimeIndex(rebal_dates))

    for date in all_dates:
        # Mark-to-market
        port_value = cash
        for tk, pos in holdings.items():
            if tk in close.columns and not pd.isna(close.loc[date, tk]):
                port_value += pos['shares'] * close.loc[date, tk]

        regime = get_regime(spy_close, date)
        equity_curve.append({'date': date, 'equity': port_value, 'regime': regime})

        # Rebalance on month-end dates
        if date in rebal_set:
            # Get new selections
            new_selections = selection_fn(close, date, holdings)
            new_tickers = set(t for t, w in new_selections)

            # Sell positions not in new selections
            for tk in list(holdings.keys()):
                if tk not in new_tickers:
                    if tk in close.columns and not pd.isna(close.loc[date, tk]):
                        sell_price = apply_slippage(close.loc[date, tk], 'sell')
                        pnl = (sell_price - holdings[tk]['entry_price']) * holdings[tk]['shares']
                        cash += holdings[tk]['shares'] * sell_price
                        hold_days = (date - holdings[tk]['entry_date']).days
                        trades.append({
                            'ticker': tk,
                            'entry_date': holdings[tk]['entry_date'],
                            'exit_date': date,
                            'entry_price': holdings[tk]['entry_price'],
                            'exit_price': sell_price,
                            'shares': holdings[tk]['shares'],
                            'pnl': pnl,
                            'hold_days': hold_days,
                            'regime': regime
                        })
                        del holdings[tk]

            # Buy new positions
            for tk, weight in new_selections:
                if tk not in holdings and weight > 0:
                    if tk in close.columns and not pd.isna(close.loc[date, tk]):
                        buy_price = apply_slippage(close.loc[date, tk], 'buy')
                        alloc = port_value * weight
                        shares = int(alloc / buy_price)  # whole shares only
                        if shares > 0 and cash >= shares * buy_price:
                            cash -= shares * buy_price
                            holdings[tk] = {
                                'shares': shares,
                                'entry_price': buy_price,
                                'entry_date': date
                            }

    # Close remaining positions at end
    final_date = all_dates[-1]
    for tk in list(holdings.keys()):
        if tk in close.columns and not pd.isna(close.loc[final_date, tk]):
            sell_price = apply_slippage(close.loc[final_date, tk], 'sell')
            pnl = (sell_price - holdings[tk]['entry_price']) * holdings[tk]['shares']
            cash += holdings[tk]['shares'] * sell_price
            hold_days = (final_date - holdings[tk]['entry_date']).days
            regime = get_regime(spy_close, final_date)
            trades.append({
                'ticker': tk,
                'entry_date': holdings[tk]['entry_date'],
                'exit_date': final_date,
                'entry_price': holdings[tk]['entry_price'],
                'exit_price': sell_price,
                'shares': holdings[tk]['shares'],
                'pnl': pnl,
                'hold_days': hold_days,
                'regime': regime
            })

    return trades, equity_curve


def compute_metrics(trades, equity_curve):
    """Compute all required metrics from trades and equity curve."""
    if not trades:
        return empty_metrics()

    df = pd.DataFrame(equity_curve)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')

    # Daily returns
    daily_ret = df['equity'].pct_change().dropna()

    # Annualized Sharpe
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
    cum = df['equity']
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min() * 100

    # Trade-level metrics
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / len(pnls) if pnls else 0
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_return = (df['equity'].iloc[-1] - CAPITAL) / CAPITAL * 100
    avg_hold = np.mean([t['hold_days'] for t in trades]) if trades else 0

    # Regime breakdown
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']

    bull_pnls = [t['pnl'] for t in bull_trades]
    bear_pnls = [t['pnl'] for t in bear_trades]

    # Regime sharpe from trade returns (approximate)
    def trade_sharpe(trade_list):
        if len(trade_list) < 2:
            return 0.0
        rets = [t['pnl'] / (t['entry_price'] * t['shares']) for t in trade_list if t['shares'] > 0]
        if len(rets) < 2 or np.std(rets) == 0:
            return 0.0
        # Annualize: ~12 monthly rebalances per year
        return (np.mean(rets) / np.std(rets)) * np.sqrt(12)

    bull_sharpe = trade_sharpe(bull_trades)
    bear_sharpe = trade_sharpe(bear_trades)

    # Regime gap: |bull_sharpe - bear_sharpe| / max(|bull|, |bear|)
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0.0

    return {
        'total_return_pct': round(total_return, 2),
        'final_equity': round(df['equity'].iloc[-1], 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'win_rate': round(win_rate, 4),
        'num_trades': len(trades),
        'max_drawdown_pct': round(max_dd, 2),
        'avg_hold_days': round(avg_hold, 1),
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
    }


def empty_metrics():
    return {k: 0 for k in ['total_return_pct', 'final_equity', 'sharpe', 'sortino',
                            'profit_factor', 'win_rate', 'num_trades', 'max_drawdown_pct',
                            'avg_hold_days', 'bull_trades', 'bear_trades', 'bull_sharpe',
                            'bear_sharpe', 'regime_gap']}


def permutation_test(trades, n_iter=PERM_ITERATIONS):
    """
    Shuffle which stocks get selected each month.
    Compare actual mean PnL to shuffled distribution.
    """
    if not trades or len(trades) < 5:
        return {'perm_p_value': 1.0, 'actual_mean_pnl': 0.0, 'perm_mean_pnl': 0.0}

    pnls = np.array([t['pnl'] for t in trades])
    actual_mean = pnls.mean()

    count_better = 0
    perm_means = []
    for _ in range(n_iter):
        shuffled = np.random.permutation(pnls)
        # Take random subset of same size (simulates random stock selection)
        k = max(1, len(pnls) // 2)
        sample_mean = shuffled[:k].mean()
        perm_means.append(sample_mean)
        if sample_mean >= actual_mean:
            count_better += 1

    p_value = count_better / n_iter
    return {
        'perm_p_value': round(p_value, 4),
        'actual_mean_pnl': round(actual_mean, 2),
        'perm_mean_pnl': round(np.mean(perm_means), 2),
    }


def five_gate_check(metrics, perm_result):
    """Apply the 5-gate validation."""
    g1 = metrics['sharpe'] > 0.5
    g2 = perm_result['perm_p_value'] < 0.05
    g3 = metrics['regime_gap'] < 0.5
    g4 = metrics['max_drawdown_pct'] > -50.0
    g5 = metrics['num_trades'] >= 20

    return {
        'G1_sharpe_gt_0.5': {'value': metrics['sharpe'], 'passed': g1},
        'G2_perm_p_lt_0.05': {'value': perm_result['perm_p_value'], 'passed': g2},
        'G3_regime_gap_lt_0.5': {'value': metrics['regime_gap'], 'passed': g3},
        'G4_maxdd_gt_neg50': {'value': metrics['max_drawdown_pct'], 'passed': g4},
        'G5_trades_gte_20': {'value': metrics['num_trades'], 'passed': g5},
        'all_passed': g1 and g2 and g3 and g4 and g5
    }


# ─── VARIANT SELECTION FUNCTIONS ─────────────────────────────────

def make_variant_a(close):
    """A) Classic 12-1 Momentum: 252d return minus 21d return, top 3 growth."""
    ret_252 = compute_returns(close[GROWTH], 252)
    ret_21 = compute_returns(close[GROWTH], 21)
    mom_12_1 = ret_252 - ret_21

    def select(close_df, date, holdings):
        if date not in mom_12_1.index:
            return []
        scores = mom_12_1.loc[date].dropna()
        if len(scores) < 3:
            return []
        top3 = scores.nlargest(3).index.tolist()
        w = 1.0 / NUM_POSITIONS
        return [(t, w) for t in top3]

    return select


def make_variant_b(close):
    """B) Triple Momentum Score: 21d/63d/126d weighted 20/40/40."""
    ret_21 = compute_returns(close[GROWTH], 21)
    ret_63 = compute_returns(close[GROWTH], 63)
    ret_126 = compute_returns(close[GROWTH], 126)
    composite = 0.2 * ret_21 + 0.4 * ret_63 + 0.4 * ret_126

    def select(close_df, date, holdings):
        if date not in composite.index:
            return []
        scores = composite.loc[date].dropna()
        if len(scores) < 3:
            return []
        top3 = scores.nlargest(3).index.tolist()
        w = 1.0 / NUM_POSITIONS
        return [(t, w) for t in top3]

    return select


def make_variant_c(close):
    """C) Momentum + Mean Reversion: 126d mom for selection, RSI<40 for entry."""
    ret_126 = compute_returns(close[GROWTH], 126)
    # Precompute RSI for all growth stocks
    rsi_dict = {}
    for tk in GROWTH:
        rsi_dict[tk] = compute_rsi(close[tk], period=5)

    def select(close_df, date, holdings):
        if date not in ret_126.index:
            return []
        scores = ret_126.loc[date].dropna()
        if len(scores) < 3:
            return []
        top3 = scores.nlargest(3).index.tolist()
        w = 1.0 / NUM_POSITIONS
        selections = []
        for tk in top3:
            if tk in rsi_dict and date in rsi_dict[tk].index:
                rsi_val = rsi_dict[tk].loc[date]
                if not pd.isna(rsi_val) and rsi_val < 40:
                    selections.append((tk, w))
                # else: hold cash for this slot (don't buy if RSI too high)
        return selections

    return select


def make_variant_d(close):
    """D) Acceleration Momentum: 3m return minus 6m return (annualized)."""
    ret_63 = compute_returns(close[GROWTH], 63)
    ret_126 = compute_returns(close[GROWTH], 126)
    # Annualize: 63d ~ 0.25yr, 126d ~ 0.5yr
    ann_63 = (1 + ret_63) ** (252/63) - 1
    ann_126 = (1 + ret_126) ** (252/126) - 1
    acceleration = ann_63 - ann_126

    def select(close_df, date, holdings):
        if date not in acceleration.index:
            return []
        scores = acceleration.loc[date].dropna()
        # Filter to positive acceleration only
        scores = scores[np.isfinite(scores)]
        if len(scores) < 3:
            return []
        top3 = scores.nlargest(3).index.tolist()
        w = 1.0 / NUM_POSITIONS
        return [(t, w) for t in top3]

    return select


def make_variant_e(close):
    """E) Momentum with Crash Stop: 12-1 mom, but go to GLD if any stock drops >15% in a month."""
    ret_252 = compute_returns(close[GROWTH], 252)
    ret_21 = compute_returns(close[GROWTH], 21)
    mom_12_1 = ret_252 - ret_21

    cooldown_until = [None]  # mutable container for closure

    def select(close_df, date, holdings):
        # Check cooldown
        if cooldown_until[0] is not None and date <= cooldown_until[0]:
            return [('GLD', 1.0)]

        if date not in mom_12_1.index:
            return []

        scores = mom_12_1.loc[date].dropna()
        if len(scores) < 3:
            return []
        top3 = scores.nlargest(3).index.tolist()

        # Check if any of our held stocks crashed >15% this month
        ret_1m = compute_returns(close_df[GROWTH], 21)
        if date in ret_1m.index:
            for tk in top3:
                if tk in ret_1m.columns:
                    monthly_ret = ret_1m.loc[date, tk]
                    if not pd.isna(monthly_ret) and monthly_ret < -0.15:
                        # Crash detected - go to GLD for 1 month
                        cooldown_until[0] = date + pd.Timedelta(days=30)
                        return [('GLD', 1.0)]

        w = 1.0 / NUM_POSITIONS
        return [(t, w) for t in top3]

    return select


def make_variant_f(close):
    """F) Adaptive Lookback: VIX-based lookback selection."""
    # Download VIX
    print("  Downloading VIX data...")
    vix = yf.download('^VIX', start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)['Close']

    ret_21 = compute_returns(close[GROWTH], 21)
    ret_63 = compute_returns(close[GROWTH], 63)
    ret_126 = compute_returns(close[GROWTH], 126)

    def select(close_df, date, holdings):
        # Get VIX level
        vix_dates = vix.index[vix.index <= date]
        if len(vix_dates) == 0:
            return []
        vix_val = vix.loc[vix_dates[-1]]
        if isinstance(vix_val, pd.Series):
            vix_val = vix_val.iloc[0]
        if pd.isna(vix_val):
            vix_val = 20  # default mid

        # Select lookback based on VIX
        if vix_val < 18:
            scores = ret_126.loc[date] if date in ret_126.index else pd.Series(dtype=float)
        elif vix_val <= 25:
            scores = ret_63.loc[date] if date in ret_63.index else pd.Series(dtype=float)
        else:
            scores = ret_21.loc[date] if date in ret_21.index else pd.Series(dtype=float)

        scores = scores.dropna()
        if len(scores) < 3:
            return []
        top3 = scores.nlargest(3).index.tolist()
        w = 1.0 / NUM_POSITIONS
        return [(t, w) for t in top3]

    return select


def run_all():
    """Run all variants and compile results."""
    close = download_data()
    print(f"Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")

    variants_config = {
        'A_classic_12_1': {
            'fn_maker': make_variant_a,
            'desc': 'Classic 12-1 Momentum: Buy top-3 growth stocks by (252d - 21d) return. Monthly rebalance. Academic baseline.',
        },
        'B_triple_momentum': {
            'fn_maker': make_variant_b,
            'desc': 'Triple Momentum Score: Composite of 1m/3m/6m returns (20/40/40 weights). Top-3 growth. Monthly rebalance.',
        },
        'C_mom_mean_revert': {
            'fn_maker': make_variant_c,
            'desc': 'Momentum + Mean Reversion: 6m momentum selects stocks, 5d RSI<40 gates entry. Buy dips in uptrends.',
        },
        'D_acceleration': {
            'fn_maker': make_variant_d,
            'desc': 'Acceleration Momentum: Rank by (3m annualized - 6m annualized) return. Stocks accelerating upward.',
        },
        'E_crash_stop': {
            'fn_maker': make_variant_e,
            'desc': 'Momentum + Crash Stop: 12-1 momentum but flee to GLD for 1 month if any top-3 stock drops >15%.',
        },
        'F_adaptive_lookback': {
            'fn_maker': make_variant_f,
            'desc': 'Adaptive Lookback: VIX<18 uses 126d, VIX 18-25 uses 63d, VIX>25 uses 21d momentum. Top-3 growth.',
        },
    }

    results = {
        'strategy': 'Multi-Timeframe Momentum',
        'run_date': datetime.now().isoformat(),
        'oot_period': f'{OOT_START} to {OOT_END}',
        'capital': CAPITAL,
        'variants': {}
    }

    for vname, vcfg in variants_config.items():
        print(f"\n{'='*60}")
        print(f"Running {vname}: {vcfg['desc'][:80]}...")

        select_fn = vcfg['fn_maker'](close)
        trades, equity_curve = run_backtest(close, select_fn, vname)

        metrics = compute_metrics(trades, equity_curve)
        perm = permutation_test(trades)
        gates = five_gate_check(metrics, perm)

        print(f"  Trades: {metrics['num_trades']}, Sharpe: {metrics['sharpe']:.3f}, "
              f"Return: {metrics['total_return_pct']:.1f}%, MaxDD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Perm p={perm['perm_p_value']:.4f}, Regime gap={metrics['regime_gap']:.3f}")
        print(f"  5-Gate: {'PASS' if gates['all_passed'] else 'FAIL'} "
              f"[{''.join('Y' if gates[g]['passed'] else 'N' for g in gates if g != 'all_passed')}]")

        results['variants'][vname] = {
            'description': vcfg['desc'],
            'metrics': metrics,
            'regime': {
                'bull_trades': metrics['bull_trades'],
                'bear_trades': metrics['bear_trades'],
                'bull_sharpe': metrics['bull_sharpe'],
                'bear_sharpe': metrics['bear_sharpe'],
                'regime_gap': metrics['regime_gap'],
            },
            'permutation': perm,
            'gates': gates,
        }

    # Save results
    output_path = '/home/jupiter/Lvl3Quant/data/multi_timeframe_momentum_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Variant':<25} {'Sharpe':>7} {'Sort':>7} {'Return%':>8} {'MaxDD%':>7} {'WR':>6} {'#Tr':>5} {'PF':>6} {'Perm':>6} {'RGap':>6} {'5G':>4}")
    print('-'*80)
    for vname, vdata in results['variants'].items():
        m = vdata['metrics']
        p = vdata['permutation']
        g = vdata['gates']
        label = vname[:24]
        print(f"{label:<25} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} {m['total_return_pct']:>7.1f}% "
              f"{m['max_drawdown_pct']:>6.1f}% {m['win_rate']:>5.1%} {m['num_trades']:>5} "
              f"{m['profit_factor']:>6.2f} {p['perm_p_value']:>5.3f} {m['regime_gap']:>6.3f} "
              f"{'PASS' if g['all_passed'] else 'FAIL':>4}")

    return results


if __name__ == '__main__':
    results = run_all()
