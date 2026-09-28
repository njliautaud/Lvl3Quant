#!/usr/bin/env python3
"""
Mean Reversion with Stop Losses Backtest
Tests whether adding stop losses to QMR-A (quality mean reversion) reduces drawdowns
without killing the edge. 6 variants: no stop, 3%/5%/8% fixed stops, trailing stop,
dynamic ATR stop.

5-gate validation: Sharpe>0.5, permutation p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
TICKERS = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
SPY = 'SPY'
START_DATE = '2021-01-01'  # extra lookback for indicators
OOT_START = '2022-01-01'
OOT_END = '2026-07-31'
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
MAX_POSITIONS = 3
MAX_PER_TRADE = 200.0
HOLD_DAYS = 10
N_PERMUTATIONS = 1000

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/mr_with_stops_results.json'


def download_data():
    """Download all price data including high/low for intraday stop checks."""
    all_tickers = TICKERS + [SPY]
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    close = data['Close'].ffill().dropna(how='all')
    high = data['High'].ffill().dropna(how='all')
    low = data['Low'].ffill().dropna(how='all')
    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


def compute_rsi(series, period=14):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    """Compute Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_indicators(close, high, low):
    """Pre-compute all indicators needed for signals."""
    indicators = {}

    for t in TICKERS:
        if t not in close.columns:
            continue
        s = close[t]
        indicators[t] = {
            'close': s,
            'high': high[t],
            'low': low[t],
            'rsi14': compute_rsi(s, 14),
            'high20': s.rolling(20).max(),
            'atr14': compute_atr(high[t], low[t], s, 14),
        }

    # SPY 200-SMA for regime
    indicators['SPY_200SMA'] = close[SPY].rolling(200).mean()
    indicators['SPY_close'] = close[SPY]

    return indicators


def get_regime(date, indicators):
    """Bull if SPY > 200-SMA, else Bear."""
    spy_close = indicators['SPY_close']
    spy_sma = indicators['SPY_200SMA']
    if date in spy_close.index and date in spy_sma.index:
        if pd.notna(spy_close.loc[date]) and pd.notna(spy_sma.loc[date]):
            return 'bull' if spy_close.loc[date] > spy_sma.loc[date] else 'bear'
    return 'unknown'


def generate_signals(date, indicators):
    """QMR-A: Buy when stock drops >5% from 20-day high AND RSI(14) < 35."""
    signals = []
    for t in TICKERS:
        if t not in indicators or 'close' not in indicators[t]:
            continue
        ind = indicators[t]
        if date not in ind['close'].index:
            continue
        price = ind['close'].get(date)
        high20 = ind['high20'].get(date)
        rsi = ind['rsi14'].get(date)
        if pd.isna(price) or pd.isna(high20) or pd.isna(rsi):
            continue
        drawdown = (price - high20) / high20
        if drawdown < -0.05 and rsi < 35:
            signals.append((t, price))
    return signals


# ── Stop Loss Logic ────────────────────────────────────────────────────────

def check_stop_none(pos, date_low, date_high, date_close, indicators):
    """No stop loss."""
    return False, None

def check_stop_fixed_3pct(pos, date_low, date_high, date_close, indicators):
    """3% fixed stop from entry."""
    stop_price = pos['entry_price'] * (1 - 0.03)
    if date_low <= stop_price:
        return True, stop_price
    return False, None

def check_stop_fixed_5pct(pos, date_low, date_high, date_close, indicators):
    """5% fixed stop from entry."""
    stop_price = pos['entry_price'] * (1 - 0.05)
    if date_low <= stop_price:
        return True, stop_price
    return False, None

def check_stop_fixed_8pct(pos, date_low, date_high, date_close, indicators):
    """8% fixed stop from entry."""
    stop_price = pos['entry_price'] * (1 - 0.08)
    if date_low <= stop_price:
        return True, stop_price
    return False, None

def check_stop_trailing_3pct(pos, date_low, date_high, date_close, indicators):
    """3% trailing stop from peak since entry."""
    # Update peak
    if date_high > pos.get('peak_price', pos['entry_price']):
        pos['peak_price'] = date_high
    stop_price = pos['peak_price'] * (1 - 0.03)
    if date_low <= stop_price:
        return True, stop_price
    return False, None

def check_stop_dynamic_atr(pos, date_low, date_high, date_close, indicators):
    """Dynamic stop at 1.5x ATR(14) below entry."""
    t = pos['ticker']
    if t not in indicators or 'atr14' not in indicators[t]:
        return False, None
    date = pos.get('current_date')
    if date is None:
        return False, None
    atr = indicators[t]['atr14'].get(date)
    if pd.isna(atr):
        return False, None
    stop_price = pos['entry_price'] - 1.5 * atr
    if date_low <= stop_price:
        return True, max(stop_price, 0.01)
    return False, None


VARIANTS = {
    'A': {'stop_fn': check_stop_none, 'desc': 'No stop loss, fixed 10-day exit (BASELINE)'},
    'B': {'stop_fn': check_stop_fixed_3pct, 'desc': '3% stop loss from entry, max 10-day hold'},
    'C': {'stop_fn': check_stop_fixed_5pct, 'desc': '5% stop loss from entry, max 10-day hold'},
    'D': {'stop_fn': check_stop_fixed_8pct, 'desc': '8% stop loss from entry, max 10-day hold'},
    'E': {'stop_fn': check_stop_trailing_3pct, 'desc': 'Trailing stop 3% from peak, max 10-day hold'},
    'F': {'stop_fn': check_stop_dynamic_atr, 'desc': 'Dynamic stop 1.5x ATR(14) below entry, max 10-day hold'},
}


def run_backtest(close, high, low, indicators, stop_fn):
    """Run a single variant backtest with stop loss logic."""
    oot_dates = close.loc[OOT_START:OOT_END].index
    if len(oot_dates) == 0:
        return [], pd.Series(dtype=float)

    capital = STARTING_CAPITAL
    positions = []
    trades = []
    equity = {}

    for date in oot_dates:
        # ── Check stops and time exits ──
        still_open = []
        for pos in positions:
            ticker = pos['ticker']
            # Get today's price data for this ticker
            if date not in close.index:
                still_open.append(pos)
                continue

            cur_close = close.loc[date, ticker] if ticker in close.columns else np.nan
            cur_high = high.loc[date, ticker] if ticker in high.columns else np.nan
            cur_low = low.loc[date, ticker] if ticker in low.columns else np.nan

            if pd.isna(cur_close) or pd.isna(cur_high) or pd.isna(cur_low):
                still_open.append(pos)
                continue

            pos['current_date'] = date

            # Check stop loss first (higher priority)
            stopped, stop_price = stop_fn(pos, float(cur_low), float(cur_high), float(cur_close), indicators)

            if stopped:
                # Stopped out - exit at stop price with slippage
                exit_price = stop_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                capital += exit_price * pos['shares']
                ret = (exit_price / pos['entry_price']) - 1
                trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'entry_price': float(pos['entry_price']),
                    'exit_price': float(exit_price),
                    'shares': int(pos['shares']),
                    'pnl': float(pnl),
                    'return': float(ret),
                    'regime': pos['regime'],
                    'exit_reason': 'stop',
                })
            elif date >= pos['exit_target']:
                # Time exit
                exit_price = float(cur_close) * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                capital += exit_price * pos['shares']
                ret = (exit_price / pos['entry_price']) - 1
                trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'entry_price': float(pos['entry_price']),
                    'exit_price': float(exit_price),
                    'shares': int(pos['shares']),
                    'pnl': float(pnl),
                    'return': float(ret),
                    'regime': pos['regime'],
                    'exit_reason': 'time',
                })
            else:
                still_open.append(pos)

        positions = still_open

        # ── Generate new entry signals ──
        if len(positions) < MAX_POSITIONS:
            signals = generate_signals(date, indicators)
            held_tickers = {p['ticker'] for p in positions}
            signals = [(t, p) for t, p in signals if t not in held_tickers]

            slots = MAX_POSITIONS - len(positions)
            for t, price in signals[:slots]:
                entry_price = price * (1 + SLIPPAGE_PCT)
                alloc = min(MAX_PER_TRADE, capital * 0.95)
                if alloc < 10 or capital < 10:
                    continue
                shares = int(alloc / entry_price)
                if shares < 1:
                    continue
                cost = shares * entry_price
                capital -= cost
                regime = get_regime(date, indicators)
                exit_idx = min(oot_dates.get_loc(date) + HOLD_DAYS, len(oot_dates) - 1)
                exit_target = oot_dates[exit_idx]
                positions.append({
                    'ticker': t,
                    'entry_price': entry_price,
                    'entry_date': date,
                    'shares': shares,
                    'exit_target': exit_target,
                    'regime': regime,
                    'peak_price': entry_price,  # for trailing stop
                    'current_date': date,
                })

        # ── Mark-to-market ──
        mtm = capital
        for pos in positions:
            curr = close.loc[date, pos['ticker']] if pos['ticker'] in close.columns else pos['entry_price']
            if pd.isna(curr):
                curr = pos['entry_price']
            mtm += float(curr) * pos['shares']
        equity[date] = mtm

    # Close remaining positions
    last_date = oot_dates[-1]
    for pos in positions:
        t = pos['ticker']
        exit_price_raw = close.loc[last_date, t] if t in close.columns else pos['entry_price']
        if pd.isna(exit_price_raw):
            exit_price_raw = pos['entry_price']
        exit_price = float(exit_price_raw) * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        capital += exit_price * pos['shares']
        ret = (exit_price / pos['entry_price']) - 1
        trades.append({
            'ticker': t,
            'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': last_date.strftime('%Y-%m-%d'),
            'entry_price': float(pos['entry_price']),
            'exit_price': float(exit_price),
            'shares': int(pos['shares']),
            'pnl': float(pnl),
            'return': float(ret),
            'regime': pos['regime'],
            'exit_reason': 'end',
        })

    equity_series = pd.Series(equity).sort_index()
    return trades, equity_series


def compute_metrics(trades, equity_series):
    """Compute performance metrics."""
    if len(trades) == 0:
        return {
            'n_trades': 0, 'total_pnl': 0, 'total_return_pct': 0,
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
            'max_drawdown_pct': 0, 'avg_winner_pct': 0, 'avg_loser_pct': 0,
            'risk_reward_ratio': 0, 'pct_stopped_out': 0,
        }

    total_pnl = sum(t['pnl'] for t in trades)
    total_return = (equity_series.iloc[-1] / STARTING_CAPITAL - 1) if len(equity_series) > 0 else 0

    # Daily returns
    if len(equity_series) > 1:
        daily_rets = equity_series.pct_change().dropna()
        ann_factor = np.sqrt(252)
        sharpe = (daily_rets.mean() / daily_rets.std() * ann_factor) if daily_rets.std() > 0 else 0
        downside = daily_rets[daily_rets < 0].std()
        sortino = (daily_rets.mean() / downside * ann_factor) if downside > 0 else 0
    else:
        sharpe = sortino = 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win rate
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(winners) / len(trades) if trades else 0

    # Average winner vs loser
    avg_winner = np.mean([t['return'] for t in winners]) * 100 if winners else 0
    avg_loser = np.mean([t['return'] for t in losers]) * 100 if losers else 0
    risk_reward = abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf')

    # Max drawdown
    if len(equity_series) > 0:
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
    else:
        max_dd = 0

    # Stop-out rate
    stopped = sum(1 for t in trades if t.get('exit_reason') == 'stop')
    pct_stopped = stopped / len(trades) if trades else 0

    return {
        'n_trades': len(trades),
        'total_pnl': round(total_pnl, 2),
        'total_return_pct': round(total_return * 100, 2),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(profit_factor), 3),
        'win_rate': round(float(win_rate), 3),
        'max_drawdown_pct': round(float(max_dd) * 100, 2),
        'avg_winner_pct': round(float(avg_winner), 3),
        'avg_loser_pct': round(float(avg_loser), 3),
        'risk_reward_ratio': round(float(risk_reward), 3),
        'pct_stopped_out': round(float(pct_stopped) * 100, 1),
    }


def compute_regime_sharpe(trades):
    """Compute Sharpe for bull and bear trades separately."""
    bull_rets = [t['return'] for t in trades if t['regime'] == 'bull']
    bear_rets = [t['return'] for t in trades if t['regime'] == 'bear']

    def _sharpe(rets):
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        if arr.std() == 0:
            return 0.0
        return float(arr.mean() / arr.std() * np.sqrt(25))

    s_bull = _sharpe(bull_rets)
    s_bear = _sharpe(bear_rets)

    denom = max(abs(s_bull), abs(s_bear))
    regime_gap = abs(s_bull - s_bear) / denom if denom > 0 else 0.0

    return {
        'sharpe_bull': round(s_bull, 3),
        'sharpe_bear': round(s_bear, 3),
        'n_bull': len(bull_rets),
        'n_bear': len(bear_rets),
        'regime_gap': round(regime_gap, 3),
    }


def permutation_test(trades, equity_series, n_perms=N_PERMUTATIONS):
    """Shuffle return signs to test statistical significance."""
    if len(trades) < 5 or len(equity_series) < 10:
        return 1.0

    # Observed Sharpe from daily returns
    observed_sharpe = 0
    if len(equity_series) > 1:
        daily_rets = equity_series.pct_change().dropna()
        if daily_rets.std() > 0:
            observed_sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252)

    trade_returns = np.array([t['return'] for t in trades])
    n_better = 0
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(trade_returns))
        scrambled = trade_returns * signs
        if scrambled.std() > 0:
            perm_sharpe = scrambled.mean() / scrambled.std() * np.sqrt(25)
        else:
            perm_sharpe = 0
        if perm_sharpe >= observed_sharpe:
            n_better += 1

    return round(n_better / n_perms, 4)


def five_gate_validation(metrics, regime_metrics, perm_p):
    """Run 5-gate validation."""
    gates = {
        'G1_sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'G2_perm_p_lt_0.05': perm_p < 0.05,
        'G3_regime_gap_lt_0.5': regime_metrics['regime_gap'] < 0.5,
        'G4_maxdd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'G5_min_20_trades': metrics['n_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


def main():
    close, high, low = download_data()
    indicators = compute_indicators(close, high, low)

    results = {}
    summary_rows = []

    for variant_name, cfg in VARIANTS.items():
        print(f"\n{'='*70}")
        print(f"Variant {variant_name}: {cfg['desc']}")
        print(f"{'='*70}")

        trades, equity = run_backtest(close, high, low, indicators, cfg['stop_fn'])
        metrics = compute_metrics(trades, equity)
        regime = compute_regime_sharpe(trades)
        perm_p = permutation_test(trades, equity)
        gates = five_gate_validation(metrics, regime, perm_p)

        print(f"  Trades: {metrics['n_trades']}, PnL: ${metrics['total_pnl']:.2f}, "
              f"Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, "
              f"PF: {metrics['profit_factor']:.2f}, WR: {metrics['win_rate']:.1%}")
        print(f"  MaxDD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Avg Winner: {metrics['avg_winner_pct']:.2f}%, Avg Loser: {metrics['avg_loser_pct']:.2f}%, "
              f"Risk/Reward: {metrics['risk_reward_ratio']:.2f}")
        print(f"  Stopped Out: {metrics['pct_stopped_out']:.1f}%")
        print(f"  Regime: Bull={regime['sharpe_bull']:.3f} ({regime['n_bull']}), "
              f"Bear={regime['sharpe_bear']:.3f} ({regime['n_bear']}), Gap={regime['regime_gap']:.3f}")
        print(f"  Perm p-value: {perm_p:.4f}")
        gate_str = ' | '.join(f'{k}={v}' for k, v in gates.items() if k != 'all_pass')
        passed = 'PASS' if gates['all_pass'] else 'FAIL'
        print(f"  Gates: {gate_str}")
        print(f"  >>> 5-GATE: {passed}")

        results[variant_name] = {
            'description': cfg['desc'],
            'hold_days': HOLD_DAYS,
            'metrics': metrics,
            'regime': regime,
            'permutation_p_value': perm_p,
            'gates': {k: bool(v) for k, v in gates.items()},
            'sample_trades': trades[:5] if len(trades) > 5 else trades,
        }

        summary_rows.append({
            'Variant': variant_name,
            'Desc': cfg['desc'][:40],
            'Trades': metrics['n_trades'],
            'Return': f"{metrics['total_return_pct']:.1f}%",
            'Sharpe': metrics['sharpe'],
            'Sortino': metrics['sortino'],
            'MaxDD': f"{metrics['max_drawdown_pct']:.1f}%",
            'WR': f"{metrics['win_rate']:.1%}",
            'PF': metrics['profit_factor'],
            'AvgWin': f"{metrics['avg_winner_pct']:.2f}%",
            'AvgLoss': f"{metrics['avg_loser_pct']:.2f}%",
            'R:R': f"{metrics['risk_reward_ratio']:.2f}",
            'Stopped%': f"{metrics['pct_stopped_out']:.1f}%",
            'RegimeGap': regime['regime_gap'],
            'Perm_p': perm_p,
            '5Gate': passed,
        })

    # ── Print comparison table ──
    print(f"\n\n{'='*120}")
    print("COMPARISON TABLE: Mean Reversion Stop Loss Variants")
    print(f"{'='*120}")
    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    # ── Key analysis ──
    print(f"\n\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")

    baseline = results.get('A', {}).get('metrics', {})
    if baseline:
        bl_sharpe = baseline.get('sharpe', 0)
        bl_mdd = baseline.get('max_drawdown_pct', 0)
        print(f"\nBaseline (A - no stop): Sharpe={bl_sharpe:.3f}, MaxDD={bl_mdd:.1f}%")
        print()
        for v in ['B', 'C', 'D', 'E', 'F']:
            if v not in results:
                continue
            m = results[v]['metrics']
            sharpe_delta = m['sharpe'] - bl_sharpe
            mdd_delta = m['max_drawdown_pct'] - bl_mdd
            direction = "IMPROVED" if sharpe_delta > 0 else "HURT"
            mdd_dir = "REDUCED" if mdd_delta > 0 else "WORSE"
            print(f"  {v} ({results[v]['description'][:45]}):")
            print(f"    Sharpe: {m['sharpe']:.3f} ({sharpe_delta:+.3f} vs baseline) -> {direction}")
            print(f"    MaxDD:  {m['max_drawdown_pct']:.1f}% ({mdd_delta:+.1f}pp vs baseline) -> {mdd_dir}")
            print(f"    Stopped: {m['pct_stopped_out']:.1f}%, R:R={m['risk_reward_ratio']:.2f}")
            print()

    # Best variant
    passing = {k: v for k, v in results.items() if v['gates']['all_pass']}
    if passing:
        best = max(passing.items(), key=lambda x: x[1]['metrics']['sharpe'])
        print(f"BEST PASSING VARIANT: {best[0]} — Sharpe={best[1]['metrics']['sharpe']:.3f}, "
              f"MaxDD={best[1]['metrics']['max_drawdown_pct']:.1f}%")

        # Does adding stops help?
        if best[0] == 'A':
            print("CONCLUSION: Stops HURT the mean reversion edge. Best to let trades run.")
        else:
            print(f"CONCLUSION: {best[0]} stop improves risk-adjusted returns vs no-stop baseline.")
    else:
        print("NO VARIANTS PASS 5-GATE VALIDATION")
        # Still find best Sharpe
        if results:
            best = max(results.items(), key=lambda x: x[1]['metrics']['sharpe'])
            print(f"Best (non-passing): {best[0]} — Sharpe={best[1]['metrics']['sharpe']:.3f}")

    # ── Save results ──
    # Convert for JSON serialization
    for v in results:
        for k, val in results[v]['gates'].items():
            results[v]['gates'][k] = bool(val)

    output = {
        'backtest': 'mr_with_stops',
        'description': 'Mean Reversion with Stop Losses - testing optimal stop level for QMR-A',
        'oot_period': f'{OOT_START} to {OOT_END}',
        'starting_capital': STARTING_CAPITAL,
        'entry_conditions': 'QMR-A: >5% drawdown from 20d high AND RSI(14)<35',
        'tickers': TICKERS,
        'slippage': f'{SLIPPAGE_PCT*100:.2f}% each way',
        'max_positions': MAX_POSITIONS,
        'max_per_trade': MAX_PER_TRADE,
        'hold_days': HOLD_DAYS,
        'variants': results,
        'run_timestamp': datetime.now().isoformat(),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
