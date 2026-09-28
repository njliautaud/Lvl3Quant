#!/usr/bin/env python3
"""
Overnight Gap + Quality Mean Reversion Backtest
Buy quality stocks at the CLOSE when selling off with specific overnight patterns.
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
N_PERMUTATIONS = 1000

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/overnight_gap_quality_results.json'


def download_data():
    """Download OHLCV data for all tickers."""
    all_tickers = TICKERS + [SPY]
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)

    close = data['Close'].ffill()
    opn = data['Open'].ffill()
    high = data['High'].ffill()
    low = data['Low'].ffill()
    volume = data['Volume'].ffill()

    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, opn, high, low, volume


def compute_rsi(series, period=14):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(close, opn, high, low, volume):
    """Pre-compute all indicators needed for signals."""
    indicators = {}

    for t in TICKERS:
        if t not in close.columns:
            continue
        c = close[t]
        o = opn[t]
        v = volume[t]

        # Count consecutive red days (close < prior close)
        daily_ret = c.pct_change()
        red_day = (daily_ret < 0).astype(int)
        # Rolling count of consecutive red days
        # Use a custom approach: reset counter on green day
        consec_red = pd.Series(0, index=c.index, dtype=int)
        for i in range(1, len(consec_red)):
            if red_day.iloc[i] == 1:
                consec_red.iloc[i] = consec_red.iloc[i-1] + 1
            else:
                consec_red.iloc[i] = 0

        indicators[t] = {
            'close': c,
            'open': o,
            'volume': v,
            'high20': c.rolling(20).max(),
            'low20': c.rolling(20).min(),
            'rsi14': compute_rsi(c, 14),
            'rsi5': compute_rsi(c, 5),
            'vol_avg20': v.rolling(20).mean(),
            'consec_red': consec_red,
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


def generate_signals_A(date, indicators, close):
    """Buy at close when stock is >5% below 20d high AND had 3+ red days
    AND today's close > today's open (green candle in downtrend = reversal hint)."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        ind = indicators[t]
        if date not in ind['close'].index:
            continue
        price = ind['close'].get(date)
        opn = ind['open'].get(date)
        high20 = ind['high20'].get(date)
        red_days = ind['consec_red'].get(date)
        if any(pd.isna(v) for v in [price, opn, high20]) or pd.isna(red_days):
            continue
        drawdown = (price - high20) / high20
        green_candle = price > opn
        if drawdown < -0.05 and red_days >= 3 and green_candle:
            signals.append((t, price))
    return signals


def generate_signals_B(date, indicators, close):
    """Same as A but hold 5 days instead of 1."""
    return generate_signals_A(date, indicators, close)


def generate_signals_C(date, indicators, close):
    """Buy at close when RSI(14) < 30 AND today's volume > 1.5x 20d average (capitulation selling)."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        ind = indicators[t]
        if date not in ind['close'].index:
            continue
        price = ind['close'].get(date)
        rsi = ind['rsi14'].get(date)
        vol = ind['volume'].get(date)
        vol_avg = ind['vol_avg20'].get(date)
        if any(pd.isna(v) for v in [price, rsi, vol, vol_avg]) or vol_avg == 0:
            continue
        if rsi < 30 and vol > 1.5 * vol_avg:
            signals.append((t, price))
    return signals


def generate_signals_D(date, indicators, close):
    """Buy at close when stock drops >3% intraday (close vs open) AND already down >5% from 20d high.
    Buy the capitulation."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        ind = indicators[t]
        if date not in ind['close'].index:
            continue
        price = ind['close'].get(date)
        opn = ind['open'].get(date)
        high20 = ind['high20'].get(date)
        if any(pd.isna(v) for v in [price, opn, high20]) or opn == 0:
            continue
        intraday_drop = (price - opn) / opn
        drawdown = (price - high20) / high20
        if intraday_drop < -0.03 and drawdown < -0.05:
            signals.append((t, price))
    return signals


def generate_signals_E(date, indicators, close):
    """Buy at close when 5-day RSI crosses BELOW 10 (extreme oversold)."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        ind = indicators[t]
        idx = ind['rsi5'].index
        date_loc = idx.get_loc(date) if date in idx else None
        if date_loc is None or date_loc < 1:
            continue
        price = ind['close'].get(date)
        rsi_today = ind['rsi5'].iloc[date_loc]
        rsi_yesterday = ind['rsi5'].iloc[date_loc - 1]
        if pd.isna(price) or pd.isna(rsi_today) or pd.isna(rsi_yesterday):
            continue
        # Cross below 10: yesterday >= 10 and today < 10
        if rsi_today < 10 and rsi_yesterday >= 10:
            signals.append((t, price))
    return signals


def generate_signals_F(date, indicators, close):
    """Buy at close when stock closes at 20-day low. Pure mean reversion."""
    signals = []
    for t in TICKERS:
        if t not in indicators:
            continue
        ind = indicators[t]
        if date not in ind['close'].index:
            continue
        price = ind['close'].get(date)
        low20 = ind['low20'].get(date)
        if pd.isna(price) or pd.isna(low20):
            continue
        # Close equals 20-day low (within tolerance for float comparison)
        if abs(price - low20) / price < 0.001:
            signals.append((t, price))
    return signals


VARIANTS = {
    'A': {'signal_fn': generate_signals_A, 'hold_days': 1, 'desc': '>5% below 20d high + 3+ red days + green candle, hold 1d'},
    'B': {'signal_fn': generate_signals_B, 'hold_days': 5, 'desc': '>5% below 20d high + 3+ red days + green candle, hold 5d'},
    'C': {'signal_fn': generate_signals_C, 'hold_days': 1, 'desc': 'RSI(14)<30 + volume>1.5x avg, hold 1d'},
    'D': {'signal_fn': generate_signals_D, 'hold_days': 1, 'desc': '>3% intraday drop + >5% from 20d high, hold 1d'},
    'E': {'signal_fn': generate_signals_E, 'hold_days': 3, 'desc': 'RSI(5) crosses below 10, hold 3d'},
    'F': {'signal_fn': generate_signals_F, 'hold_days': 1, 'desc': 'Close at 20-day low, hold 1d'},
}


def run_backtest(close, indicators, signal_fn, hold_days):
    """Run a single variant backtest. Returns list of trades and daily equity curve."""
    oot_dates = close.loc[OOT_START:OOT_END].index
    if len(oot_dates) == 0:
        return [], pd.Series(dtype=float)

    capital = STARTING_CAPITAL
    positions = []
    trades = []
    equity = {}

    for date in oot_dates:
        # Exit positions that hit hold period
        still_open = []
        for pos in positions:
            if date >= pos['exit_target']:
                exit_price_raw = close.loc[date, pos['ticker']] if date in close.index else pos['entry_price']
                if pd.isna(exit_price_raw):
                    still_open.append(pos)
                    continue
                exit_price = exit_price_raw * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                capital += exit_price * pos['shares']
                ret = (exit_price / pos['entry_price']) - 1
                trades.append({
                    'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                    'exit_date': date.strftime('%Y-%m-%d'),
                    'entry_price': float(pos['entry_price']),
                    'exit_price': float(exit_price),
                    'shares': int(pos['shares']),
                    'pnl': float(pnl),
                    'return': float(ret),
                    'regime': pos['regime'],
                })
            else:
                still_open.append(pos)
        positions = still_open

        # Generate signals
        if len(positions) < MAX_POSITIONS:
            signals = signal_fn(date, indicators, close)
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
                exit_idx = min(oot_dates.get_loc(date) + hold_days, len(oot_dates) - 1)
                exit_target = oot_dates[exit_idx]
                positions.append({
                    'ticker': t,
                    'entry_price': entry_price,
                    'entry_date': date,
                    'shares': shares,
                    'exit_target': exit_target,
                    'regime': regime,
                })

        # Mark-to-market
        mtm = capital
        for pos in positions:
            curr = close.loc[date, pos['ticker']] if date in close.index else pos['entry_price']
            if pd.isna(curr):
                curr = pos['entry_price']
            mtm += float(curr) * pos['shares']
        equity[date] = mtm

    # Close remaining positions
    last_date = oot_dates[-1]
    for pos in positions:
        exit_price_raw = close.loc[last_date, pos['ticker']] if last_date in close.index else pos['entry_price']
        if pd.isna(exit_price_raw):
            exit_price_raw = pos['entry_price']
        exit_price = float(exit_price_raw) * (1 - SLIPPAGE_PCT)
        pnl = (exit_price - pos['entry_price']) * pos['shares']
        capital += exit_price * pos['shares']
        ret = (exit_price / pos['entry_price']) - 1
        trades.append({
            'ticker': pos['ticker'],
            'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
            'exit_date': last_date.strftime('%Y-%m-%d'),
            'entry_price': float(pos['entry_price']),
            'exit_price': float(exit_price),
            'shares': int(pos['shares']),
            'pnl': float(pnl),
            'return': float(ret),
            'regime': pos['regime'],
        })

    equity_series = pd.Series(equity).sort_index()
    return trades, equity_series


def compute_metrics(trades, equity_series):
    """Compute performance metrics."""
    if len(trades) == 0:
        return {
            'n_trades': 0, 'total_pnl': 0, 'total_return_pct': 0,
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
            'max_drawdown_pct': 0, 'avg_trade_return_pct': 0,
        }

    trade_returns = [t['return'] for t in trades]
    total_pnl = sum(t['pnl'] for t in trades)
    total_return = (equity_series.iloc[-1] / STARTING_CAPITAL - 1) if len(equity_series) > 0 else 0

    if len(equity_series) > 1:
        daily_rets = equity_series.pct_change().dropna()
        ann_factor = np.sqrt(252)
        sharpe = (daily_rets.mean() / daily_rets.std() * ann_factor) if daily_rets.std() > 0 else 0
        downside = daily_rets[daily_rets < 0].std()
        sortino = (daily_rets.mean() / downside * ann_factor) if downside > 0 else 0
    else:
        sharpe = sortino = 0

    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    winners = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winners / len(trades)

    if len(equity_series) > 0:
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
    else:
        max_dd = 0

    avg_trade_ret = np.mean(trade_returns) * 100

    return {
        'n_trades': len(trades),
        'total_pnl': round(total_pnl, 2),
        'total_return_pct': round(total_return * 100, 2),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(profit_factor), 3),
        'win_rate': round(float(win_rate), 3),
        'max_drawdown_pct': round(float(max_dd) * 100, 2),
        'avg_trade_return_pct': round(float(avg_trade_ret), 3),
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
    """Shuffle trade return signs, recompute Sharpe. Return p-value."""
    if len(trades) < 5 or len(equity_series) < 10:
        return 1.0

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
    close, opn, high, low, volume = download_data()
    indicators = compute_indicators(close, opn, high, low, volume)

    results = {}
    summary_rows = []

    for variant_name, cfg in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Running Variant {variant_name}: {cfg['desc']}")
        print(f"{'='*60}")

        trades, equity = run_backtest(
            close, indicators,
            cfg['signal_fn'], cfg['hold_days']
        )
        metrics = compute_metrics(trades, equity)
        regime = compute_regime_sharpe(trades)
        perm_p = permutation_test(trades, equity)
        gates = five_gate_validation(metrics, regime, perm_p)

        print(f"  Trades: {metrics['n_trades']}, PnL: ${metrics['total_pnl']:.2f}, "
              f"Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, "
              f"PF: {metrics['profit_factor']:.2f}, WR: {metrics['win_rate']:.1%}")
        print(f"  MaxDD: {metrics['max_drawdown_pct']:.1f}%, AvgTrade: {metrics['avg_trade_return_pct']:.2f}%")
        print(f"  Regime: Bull Sharpe={regime['sharpe_bull']:.3f} ({regime['n_bull']}), "
              f"Bear Sharpe={regime['sharpe_bear']:.3f} ({regime['n_bear']}), Gap={regime['regime_gap']:.3f}")
        print(f"  Perm p-value: {perm_p:.4f}")
        print(f"  Gates: {' | '.join(f'{k}={v}' for k,v in gates.items())}")

        passed = 'PASS' if gates['all_pass'] else 'FAIL'
        print(f"  >>> 5-GATE: {passed}")

        results[variant_name] = {
            'description': cfg['desc'],
            'hold_days': cfg['hold_days'],
            'metrics': metrics,
            'regime': regime,
            'permutation_p_value': perm_p,
            'gates': {k: bool(v) for k, v in gates.items()},
            'sample_trades': trades[:5] if len(trades) > 5 else trades,
        }

        summary_rows.append({
            'Variant': variant_name,
            'Trades': metrics['n_trades'],
            'PnL': f"${metrics['total_pnl']:.0f}",
            'Return': f"{metrics['total_return_pct']:.1f}%",
            'Sharpe': metrics['sharpe'],
            'Sortino': metrics['sortino'],
            'PF': metrics['profit_factor'],
            'WR': f"{metrics['win_rate']:.0%}",
            'MaxDD': f"{metrics['max_drawdown_pct']:.1f}%",
            'RegGap': regime['regime_gap'],
            'PermP': perm_p,
            '5Gate': passed,
        })

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print("OVERNIGHT GAP + QUALITY MR -- SUMMARY TABLE")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL} | Slippage: {SLIPPAGE_PCT*100:.2f}%/side")
    print(f"{'='*100}")

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    passed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'PASS']
    failed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'FAIL']
    print(f"\nPASSED 5-GATE: {passed_variants if passed_variants else 'None'}")
    print(f"FAILED 5-GATE: {failed_variants if failed_variants else 'None'}")

    # ── Save results ───────────────────────────────────────────────────────
    results['_meta'] = {
        'strategy': 'Overnight Gap + Quality Mean Reversion',
        'universe': TICKERS,
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'starting_capital': STARTING_CAPITAL,
        'slippage_pct_each_way': SLIPPAGE_PCT,
        'max_positions': MAX_POSITIONS,
        'max_per_trade': MAX_PER_TRADE,
        'n_permutations': N_PERMUTATIONS,
        'run_timestamp': datetime.now().isoformat(),
        'passed_variants': passed_variants,
        'failed_variants': failed_variants,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
