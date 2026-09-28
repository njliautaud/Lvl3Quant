#!/usr/bin/env python3
"""
Composite Mean Reversion Timing Backtest
Tests whether combining multiple proven signals creates better timing than any single signal.
6 variants: RSI+Spread, Dip+Volume+Streak, ROC+RSI+Spread, Count-Based, Strict Triple, Adaptive.

5-gate validation: Sharpe>0.5, permutation p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

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
STARTING_CAPITAL = 669.0
SLIPPAGE_PCT = 0.0002  # 2bps each way
MAX_POSITIONS = 3
MAX_PER_TRADE = 200.0
DEFAULT_HOLD = 10
N_PERMUTATIONS = 1000

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/mr_timing_composite_results.json'


def download_data():
    """Download all price data."""
    all_tickers = TICKERS + [SPY]
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    close = data['Close'].ffill().dropna(how='all')
    high = data['High'].ffill().dropna(how='all')
    low = data['Low'].ffill().dropna(how='all')
    volume = data['Volume'].ffill().dropna(how='all')
    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low, volume


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(close, high, low, volume):
    """Pre-compute all indicators needed across all variants."""
    indicators = {}

    for t in TICKERS:
        if t not in close.columns:
            continue
        s = close[t]
        h = high[t]
        l = low[t]
        v = volume[t]

        hl_spread = h - l
        hl_spread_avg60 = hl_spread.rolling(60).mean()

        # Consecutive red days
        daily_ret = s.pct_change()
        red_streak = pd.Series(0, index=s.index, dtype=int)
        streak = 0
        for i in range(len(daily_ret)):
            if daily_ret.iloc[i] < 0:
                streak += 1
            else:
                streak = 0
            red_streak.iloc[i] = streak

        indicators[t] = {
            'close': s,
            'high': h,
            'low': l,
            'volume': v,
            'rsi14': compute_rsi(s, 14),
            'high20': s.rolling(20).max(),
            'vol_avg20': v.rolling(20).mean(),
            'hl_spread': hl_spread,
            'hl_spread_avg60': hl_spread_avg60,
            'roc5': s.pct_change(5) * 100,
            'roc10': s.pct_change(10) * 100,
            'red_streak': red_streak,
        }

    # SPY 200-SMA for regime
    indicators['SPY_200SMA'] = close[SPY].rolling(200).mean()
    indicators['SPY_close'] = close[SPY]

    return indicators


def get_regime(date, indicators):
    spy_close = indicators['SPY_close']
    spy_sma = indicators['SPY_200SMA']
    if date in spy_close.index and date in spy_sma.index:
        if pd.notna(spy_close.loc[date]) and pd.notna(spy_sma.loc[date]):
            return 'bull' if spy_close.loc[date] > spy_sma.loc[date] else 'bear'
    return 'unknown'


def safe_get(series, date):
    """Safely get value from series at date, returning NaN if missing."""
    if date not in series.index:
        return np.nan
    return series.loc[date]


def generate_signals_A(date, indicators):
    """Variant A: RSI + Spread Combo
    Buy when RSI(14)<35 AND HL spread < 60d avg AND >5% below 20d high. Hold 10d."""
    signals = []
    for t in TICKERS:
        if t not in indicators or 'close' not in indicators[t]:
            continue
        ind = indicators[t]
        price = safe_get(ind['close'], date)
        rsi = safe_get(ind['rsi14'], date)
        high20 = safe_get(ind['high20'], date)
        hl_spread = safe_get(ind['hl_spread'], date)
        hl_avg60 = safe_get(ind['hl_spread_avg60'], date)

        if any(pd.isna(x) for x in [price, rsi, high20, hl_spread, hl_avg60]):
            continue
        if hl_avg60 == 0:
            continue

        dd = (price - high20) / high20
        if rsi < 35 and hl_spread < hl_avg60 and dd < -0.05:
            signals.append({'ticker': t, 'price': price, 'hold': 10,
                            'score': -dd + (35 - rsi)/100})  # deeper dip + lower RSI = higher priority
    return sorted(signals, key=lambda x: -x['score'])


def generate_signals_B(date, indicators):
    """Variant B: Dip + Volume + Streak
    Buy when >7% below 20d high AND vol < 0.7x 20d avg AND 3+ red days. Hold 10d."""
    signals = []
    for t in TICKERS:
        if t not in indicators or 'close' not in indicators[t]:
            continue
        ind = indicators[t]
        price = safe_get(ind['close'], date)
        high20 = safe_get(ind['high20'], date)
        vol = safe_get(ind['volume'], date)
        vol_avg = safe_get(ind['vol_avg20'], date)
        streak = safe_get(ind['red_streak'], date)

        if any(pd.isna(x) for x in [price, high20, vol, vol_avg, streak]):
            continue
        if vol_avg == 0:
            continue

        dd = (price - high20) / high20
        vol_ratio = vol / vol_avg
        if dd < -0.07 and vol_ratio < 0.7 and streak >= 3:
            signals.append({'ticker': t, 'price': price, 'hold': 10,
                            'score': -dd})
    return sorted(signals, key=lambda x: -x['score'])


def generate_signals_C(date, indicators):
    """Variant C: ROC + RSI + Spread
    Buy when 10d ROC < -8% AND RSI < 40 AND HL spread < 60d avg. Hold 10d."""
    signals = []
    for t in TICKERS:
        if t not in indicators or 'close' not in indicators[t]:
            continue
        ind = indicators[t]
        price = safe_get(ind['close'], date)
        roc10 = safe_get(ind['roc10'], date)
        rsi = safe_get(ind['rsi14'], date)
        hl_spread = safe_get(ind['hl_spread'], date)
        hl_avg60 = safe_get(ind['hl_spread_avg60'], date)

        if any(pd.isna(x) for x in [price, roc10, rsi, hl_spread, hl_avg60]):
            continue
        if hl_avg60 == 0:
            continue

        if roc10 < -8 and rsi < 40 and hl_spread < hl_avg60:
            signals.append({'ticker': t, 'price': price, 'hold': 10,
                            'score': -roc10 + (40 - rsi)/100})
    return sorted(signals, key=lambda x: -x['score'])


def generate_signals_D(date, indicators):
    """Variant D: Count-Based Entry
    Score: +1 for each condition met (6 total). Buy when score >= 4. Hold 10d."""
    signals = []
    for t in TICKERS:
        if t not in indicators or 'close' not in indicators[t]:
            continue
        ind = indicators[t]
        price = safe_get(ind['close'], date)
        rsi = safe_get(ind['rsi14'], date)
        high20 = safe_get(ind['high20'], date)
        roc5 = safe_get(ind['roc5'], date)
        vol = safe_get(ind['volume'], date)
        vol_avg = safe_get(ind['vol_avg20'], date)
        hl_spread = safe_get(ind['hl_spread'], date)
        hl_avg60 = safe_get(ind['hl_spread_avg60'], date)
        streak = safe_get(ind['red_streak'], date)

        if any(pd.isna(x) for x in [price, rsi, high20, roc5, vol, vol_avg, hl_spread, hl_avg60, streak]):
            continue
        if high20 == 0 or vol_avg == 0 or hl_avg60 == 0:
            continue

        dd = (price - high20) / high20
        score = 0
        if rsi < 40: score += 1
        if dd < -0.05: score += 1
        if roc5 < -3: score += 1
        if vol / vol_avg < 0.8: score += 1
        if hl_spread < hl_avg60: score += 1
        if streak >= 3: score += 1

        if score >= 4:
            signals.append({'ticker': t, 'price': price, 'hold': 10, 'score': score})
    return sorted(signals, key=lambda x: -x['score'])


def generate_signals_E(date, indicators):
    """Variant E: Strict Triple Filter
    Buy ONLY when RSI < 30 AND >10% below 20d high AND HL spread < 60d avg. Hold 15d."""
    signals = []
    for t in TICKERS:
        if t not in indicators or 'close' not in indicators[t]:
            continue
        ind = indicators[t]
        price = safe_get(ind['close'], date)
        rsi = safe_get(ind['rsi14'], date)
        high20 = safe_get(ind['high20'], date)
        hl_spread = safe_get(ind['hl_spread'], date)
        hl_avg60 = safe_get(ind['hl_spread_avg60'], date)

        if any(pd.isna(x) for x in [price, rsi, high20, hl_spread, hl_avg60]):
            continue
        if hl_avg60 == 0:
            continue

        dd = (price - high20) / high20
        if rsi < 30 and dd < -0.10 and hl_spread < hl_avg60:
            signals.append({'ticker': t, 'price': price, 'hold': 15,
                            'score': -dd + (30 - rsi)/100})
    return sorted(signals, key=lambda x: -x['score'])


def generate_signals_F(date, indicators):
    """Variant F: Adaptive Threshold
    Buy when RSI < 40 AND >5% below high. Hold period adapts to dip depth."""
    signals = []
    for t in TICKERS:
        if t not in indicators or 'close' not in indicators[t]:
            continue
        ind = indicators[t]
        price = safe_get(ind['close'], date)
        rsi = safe_get(ind['rsi14'], date)
        high20 = safe_get(ind['high20'], date)

        if any(pd.isna(x) for x in [price, rsi, high20]):
            continue
        if high20 == 0:
            continue

        dd = (price - high20) / high20
        dd_pct = abs(dd) * 100

        if rsi < 40 and dd < -0.05:
            # Adaptive hold period
            if dd_pct >= 15:
                hold = 15
            elif dd_pct >= 10:
                hold = 12
            elif dd_pct >= 7:
                hold = 10
            else:
                hold = 7

            signals.append({'ticker': t, 'price': price, 'hold': hold,
                            'score': -dd + (40 - rsi)/100})
    return sorted(signals, key=lambda x: -x['score'])


VARIANTS = {
    'A_RSI_Spread': generate_signals_A,
    'B_Dip_Vol_Streak': generate_signals_B,
    'C_ROC_RSI_Spread': generate_signals_C,
    'D_CountBased': generate_signals_D,
    'E_StrictTriple': generate_signals_E,
    'F_Adaptive': generate_signals_F,
}


def run_backtest(signal_fn, close, indicators, dates):
    """Run a single variant backtest. Returns trades list and equity curve."""
    capital = STARTING_CAPITAL
    positions = []  # list of {ticker, entry_price, shares, entry_date, hold_days, exit_date_idx}
    trades = []
    equity = []

    date_list = list(dates)
    date_to_idx = {d: i for i, d in enumerate(date_list)}

    for i, date in enumerate(date_list):
        # Check exits
        new_positions = []
        for pos in positions:
            days_held = i - pos['entry_idx']
            if days_held >= pos['hold']:
                # Exit
                exit_price = safe_get(close[pos['ticker']], date)
                if pd.isna(exit_price):
                    new_positions.append(pos)
                    continue
                exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
                pnl = (exit_price_adj - pos['entry_price']) * pos['shares']
                pnl_pct = (exit_price_adj / pos['entry_price'] - 1) * 100
                capital += exit_price_adj * pos['shares']
                regime = get_regime(pos['entry_date'], indicators)
                trades.append({
                    'ticker': pos['ticker'],
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'entry_price': round(pos['entry_price'], 4),
                    'exit_price': round(exit_price_adj, 4),
                    'shares': pos['shares'],
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pnl_pct, 2),
                    'regime': regime,
                    'hold_days': days_held,
                })
            else:
                new_positions.append(pos)
        positions = new_positions

        # Check entries
        if len(positions) < MAX_POSITIONS:
            signals = signal_fn(date, indicators)
            held_tickers = {p['ticker'] for p in positions}
            for sig in signals:
                if len(positions) >= MAX_POSITIONS:
                    break
                if sig['ticker'] in held_tickers:
                    continue
                price = sig['price']
                price_adj = price * (1 + SLIPPAGE_PCT)
                invest = min(MAX_PER_TRADE, capital * 0.95)
                if invest < 10 or capital < 10:
                    continue
                shares = invest / price_adj
                if shares * price_adj > capital:
                    continue
                capital -= shares * price_adj
                positions.append({
                    'ticker': sig['ticker'],
                    'entry_price': price_adj,
                    'shares': shares,
                    'entry_date': date,
                    'entry_idx': i,
                    'hold': sig['hold'],
                })
                held_tickers.add(sig['ticker'])

        # Mark-to-market equity
        mtm = capital
        for pos in positions:
            cur_price = safe_get(close[pos['ticker']], date)
            if not pd.isna(cur_price):
                mtm += cur_price * pos['shares']
        equity.append({'date': str(date.date()), 'equity': round(mtm, 2)})

    # Force-close remaining positions at last date
    last_date = date_list[-1]
    for pos in positions:
        exit_price = safe_get(close[pos['ticker']], last_date)
        if pd.isna(exit_price):
            continue
        exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
        pnl = (exit_price_adj - pos['entry_price']) * pos['shares']
        pnl_pct = (exit_price_adj / pos['entry_price'] - 1) * 100
        capital += exit_price_adj * pos['shares']
        regime = get_regime(pos['entry_date'], indicators)
        trades.append({
            'ticker': pos['ticker'],
            'entry_date': str(pos['entry_date'].date()),
            'exit_date': str(last_date.date()),
            'entry_price': round(pos['entry_price'], 4),
            'exit_price': round(exit_price_adj, 4),
            'shares': pos['shares'],
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'regime': regime,
            'hold_days': len(date_list) - 1 - pos['entry_idx'],
        })

    return trades, equity


def compute_metrics(trades, equity):
    """Compute performance metrics from trades and equity curve."""
    if len(trades) == 0:
        return {'n_trades': 0, 'sharpe': 0, 'total_return_pct': 0, 'max_dd_pct': 0}

    pnls = [t['pnl'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    total_pnl = sum(pnls)
    total_return_pct = (total_pnl / STARTING_CAPITAL) * 100
    win_rate = len(winners) / len(pnls) * 100 if pnls else 0
    avg_win = np.mean(winners) if winners else 0
    avg_loss = np.mean(losers) if losers else 0
    profit_factor = abs(sum(winners) / sum(losers)) if losers and sum(losers) != 0 else float('inf')

    # Daily returns from equity curve
    eq_vals = [e['equity'] for e in equity]
    daily_rets = pd.Series(eq_vals).pct_change().dropna()
    sharpe = (daily_rets.mean() / daily_rets.std()) * np.sqrt(252) if daily_rets.std() > 0 else 0
    neg_rets = daily_rets[daily_rets < 0]
    sortino = (daily_rets.mean() / neg_rets.std()) * np.sqrt(252) if len(neg_rets) > 0 and neg_rets.std() > 0 else 0

    # Max drawdown from equity curve
    eq_series = pd.Series(eq_vals)
    peak = eq_series.cummax()
    dd = (eq_series - peak) / peak
    max_dd = dd.min() * 100

    # Regime analysis
    bull_pnls = [t['pnl_pct'] for t in trades if t['regime'] == 'bull']
    bear_pnls = [t['pnl_pct'] for t in trades if t['regime'] == 'bear']
    bull_sharpe = np.mean(bull_pnls) / np.std(bull_pnls) if len(bull_pnls) > 1 and np.std(bull_pnls) > 0 else 0
    bear_sharpe = np.mean(bear_pnls) / np.std(bear_pnls) if len(bear_pnls) > 1 and np.std(bear_pnls) > 0 else 0
    max_regime = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_regime if max_regime > 0 else 0

    return {
        'n_trades': len(trades),
        'n_winners': len(winners),
        'n_losers': len(losers),
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'total_return_pct': round(total_return_pct, 2),
        'avg_pnl_pct': round(np.mean(pnl_pcts), 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 999.0,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd_pct': round(max_dd, 2),
        'bull_trades': len(bull_pnls),
        'bear_trades': len(bear_pnls),
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
    }


def permutation_test(trades, equity, n_perms=N_PERMUTATIONS):
    """Shuffle trade P&Ls to test if returns are better than random."""
    if len(trades) < 5:
        return 1.0

    real_pnl = sum(t['pnl'] for t in trades)
    pnls = [t['pnl'] for t in trades]
    count_better = 0
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        shuffled = rng.permutation(pnls)
        # Simulate with shuffled P&Ls but same trade structure
        shuffled_total = sum(shuffled)
        if shuffled_total >= real_pnl:
            count_better += 1

    return round(count_better / n_perms, 4)


def validate_5gate(metrics, perm_p):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_dd_pct'] > -50,
        'trades_gte_20': metrics['n_trades'] >= 20,
    }
    gates['all_passed'] = all(gates.values())
    return gates


def main():
    print("=" * 80)
    print("COMPOSITE MEAN REVERSION TIMING BACKTEST")
    print("=" * 80)

    close, high, low, volume = download_data()
    indicators = compute_indicators(close, high, low, volume)

    # Filter to OOT period
    oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
    dates = close.index[oot_mask]
    print(f"OOT period: {dates[0].date()} to {dates[-1].date()} ({len(dates)} trading days)")
    print()

    results = {}
    for name, signal_fn in VARIANTS.items():
        print(f"--- Variant {name} ---")
        trades, equity = run_backtest(signal_fn, close, indicators, dates)
        metrics = compute_metrics(trades, equity)
        print(f"  Trades: {metrics['n_trades']}, WR: {metrics.get('win_rate', 0)}%, "
              f"Return: {metrics['total_return_pct']}%, Sharpe: {metrics['sharpe']}, "
              f"Sortino: {metrics.get('sortino', 0)}, MaxDD: {metrics['max_dd_pct']}%")

        if metrics['n_trades'] >= 5:
            perm_p = permutation_test(trades, equity)
        else:
            perm_p = 1.0
        print(f"  Permutation p-value: {perm_p}")

        gates = validate_5gate(metrics, perm_p)
        passed = sum(1 for v in gates.values() if v and isinstance(v, bool)) - (1 if gates['all_passed'] else 0)
        print(f"  Gates passed: {passed}/5, ALL: {gates['all_passed']}")

        if metrics['n_trades'] > 0:
            print(f"  Regime: Bull {metrics['bull_trades']}t (Sharpe {metrics['bull_sharpe']}), "
                  f"Bear {metrics['bear_trades']}t (Sharpe {metrics['bear_sharpe']}), Gap: {metrics['regime_gap']}")
            print(f"  PF: {metrics.get('profit_factor', 0)}, AvgWin: ${metrics['avg_win']}, AvgLoss: ${metrics['avg_loss']}")

        results[name] = {
            'metrics': metrics,
            'permutation_p': perm_p,
            'gates': gates,
            'trades': trades,
            'equity_start': equity[0]['equity'] if equity else STARTING_CAPITAL,
            'equity_end': equity[-1]['equity'] if equity else STARTING_CAPITAL,
        }
        print()

    # ── Summary ──────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY — 5-GATE VALIDATION")
    print("=" * 80)
    print(f"{'Variant':<22} {'Trades':>6} {'WR%':>6} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD%':>7} {'PF':>7} {'Perm-p':>7} {'RGap':>6} {'Pass':>5}")
    print("-" * 100)

    for name, res in results.items():
        m = res['metrics']
        p = res['permutation_p']
        g = res['gates']
        n_passed = sum(1 for k, v in g.items() if v and k != 'all_passed')
        tag = "PASS" if g['all_passed'] else f"{n_passed}/5"
        print(f"{name:<22} {m['n_trades']:>6} {m.get('win_rate',0):>5.1f}% {m['total_return_pct']:>7.1f}% "
              f"{m['sharpe']:>7.3f} {m.get('sortino',0):>8.3f} {m['max_dd_pct']:>6.1f}% "
              f"{m.get('profit_factor',0):>7.2f} {p:>7.4f} {m.get('regime_gap',0):>5.3f} {tag:>5}")

    # ── Save results ──────────────────────────────────────────────────────
    output = {
        'metadata': {
            'strategy': 'Composite Mean Reversion Timing on Quality Stocks',
            'timestamp': datetime.now().isoformat(),
            'period': f'{OOT_START} to {OOT_END}',
            'capital': STARTING_CAPITAL,
            'max_per_trade': MAX_PER_TRADE,
            'max_concurrent': MAX_POSITIONS,
            'slippage_bps': SLIPPAGE_PCT * 10000,
            'universe_size': len(TICKERS),
            'n_permutations': N_PERMUTATIONS,
        },
        'variants': {}
    }

    for name, res in results.items():
        output['variants'][name] = {
            'metrics': res['metrics'],
            'permutation_p': res['permutation_p'],
            'gates': res['gates'],
            'n_trades': res['metrics']['n_trades'],
            'sample_trades': res['trades'][:10] if res['trades'] else [],
        }

    # Custom encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
