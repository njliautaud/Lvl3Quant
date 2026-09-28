#!/usr/bin/env python3
"""
Sector Momentum + Quality Stock Overlay Backtest
Tests whether sector-level signals can IMPROVE QMR (Quality Mean Reversion) stock selection.

Insight: QMR works (Sharpe 1.03, 6/6 adversarial pass). Can we improve it by only buying
quality stocks whose SECTOR is in a favorable position?

All 6 variants start with the base QMR condition:
  - Stock drops >5% from 20-day high AND RSI(14) < 35

Then each adds a sector-level overlay filter.

5-gate validation: Sharpe>0.5, permutation p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────
TICKERS = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]

# Map each stock to its sector ETF
STOCK_SECTOR = {
    'AAPL': 'XLK', 'MSFT': 'XLK', 'AVGO': 'XLK', 'V': 'XLK', 'MA': 'XLK',
    'JPM': 'XLF',
    'JNJ': 'XLV', 'UNH': 'XLV', 'LLY': 'XLV', 'ABBV': 'XLV', 'MRK': 'XLV',
    'PG': 'XLP', 'KO': 'XLP', 'PEP': 'XLP', 'COST': 'XLP', 'WMT': 'XLP',
    'HD': 'XLY', 'AMZN': 'XLY',
    'GOOGL': 'XLC', 'META': 'XLC',
}
SECTOR_ETFS = list(set(STOCK_SECTOR.values()))  # XLK, XLF, XLV, XLP, XLY, XLC

SPY = 'SPY'
VIX_TICKER = '^VIX'
START_DATE = '2021-01-01'  # extra lookback for indicators
OOT_START = '2022-01-01'
OOT_END = '2026-07-31'
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
MAX_POSITIONS = 3
MAX_PER_TRADE = 200.0
HOLD_DAYS = 10
N_PERMUTATIONS = 1000

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/sector_momentum_timing_results.json'


def download_data():
    """Download all price data including sector ETFs and VIX."""
    stock_tickers = TICKERS + [SPY] + SECTOR_ETFS
    print(f"Downloading stock/ETF data for {len(stock_tickers)} tickers...")
    data = yf.download(stock_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    close = close.ffill().dropna(how='all')

    print(f"Downloading VIX data...")
    vix_data = yf.download(VIX_TICKER, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_close = vix_data['Close'].squeeze()
    else:
        vix_close = vix_data['Close']

    print(f"Stock data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, vix_close


def compute_rsi(series, period=14):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(close, vix_close):
    """Pre-compute all indicators needed for signals."""
    indicators = {}

    # Stock indicators
    for t in TICKERS:
        if t not in close.columns:
            continue
        s = close[t]
        indicators[t] = {
            'close': s,
            'rsi14': compute_rsi(s, 14),
            'high20': s.rolling(20).max(),
            'ret5': s.pct_change(5),
        }

    # Sector ETF indicators
    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            print(f"  WARNING: {etf} not found in data")
            continue
        s = close[etf]
        indicators[f'sector_{etf}'] = {
            'close': s,
            'rsi14': compute_rsi(s, 14),
            'ret20': s.pct_change(20),  # 20-day (1-month) return
            'ret5': s.pct_change(5),    # 5-day return
        }

    # SPY 200-SMA for regime
    indicators['SPY_200SMA'] = close[SPY].rolling(200).mean()
    indicators['SPY_close'] = close[SPY]

    # VIX
    indicators['VIX'] = vix_close

    return indicators


def get_regime(date, indicators):
    """Bull if SPY > 200-SMA, else Bear."""
    spy_close = indicators['SPY_close']
    spy_sma = indicators['SPY_200SMA']
    if date in spy_close.index and date in spy_sma.index:
        if pd.notna(spy_close.loc[date]) and pd.notna(spy_sma.loc[date]):
            return 'bull' if spy_close.loc[date] > spy_sma.loc[date] else 'bear'
    return 'unknown'


def base_qmr_filter(ticker, date, indicators):
    """Base QMR condition: stock drops >5% from 20-day high AND RSI < 35.
    Returns (True, price) if condition met, else (False, None)."""
    if ticker not in indicators:
        return False, None
    ind = indicators[ticker]
    if date not in ind['close'].index:
        return False, None
    price = ind['close'].get(date)
    high20 = ind['high20'].get(date)
    rsi = ind['rsi14'].get(date)
    if pd.isna(price) or pd.isna(high20) or pd.isna(rsi):
        return False, None
    drawdown = (price - high20) / high20
    if drawdown < -0.05 and rsi < 35:
        return True, price
    return False, None


def generate_signals_A(date, indicators, close):
    """Variant A: QMR baseline (no sector filter). CONTROL."""
    signals = []
    for t in TICKERS:
        passed, price = base_qmr_filter(t, date, indicators)
        if passed:
            signals.append((t, price))
    return signals


def generate_signals_B(date, indicators, close):
    """Variant B: QMR + sector ETF RSI < 40 (sector also oversold)."""
    signals = []
    for t in TICKERS:
        passed, price = base_qmr_filter(t, date, indicators)
        if not passed:
            continue
        sector_etf = STOCK_SECTOR[t]
        sector_key = f'sector_{sector_etf}'
        if sector_key not in indicators:
            continue
        sector_rsi = indicators[sector_key]['rsi14'].get(date)
        if pd.isna(sector_rsi):
            continue
        if sector_rsi < 40:
            signals.append((t, price))
    return signals


def generate_signals_C(date, indicators, close):
    """Variant C: QMR + sector ETF 20-day return is POSITIVE (sector recovering)."""
    signals = []
    for t in TICKERS:
        passed, price = base_qmr_filter(t, date, indicators)
        if not passed:
            continue
        sector_etf = STOCK_SECTOR[t]
        sector_key = f'sector_{sector_etf}'
        if sector_key not in indicators:
            continue
        sector_ret20 = indicators[sector_key]['ret20'].get(date)
        if pd.isna(sector_ret20):
            continue
        if sector_ret20 > 0:
            signals.append((t, price))
    return signals


def generate_signals_D(date, indicators, close):
    """Variant D: QMR + stock's 5d return < sector's 5d return (stock lagging sector)."""
    signals = []
    for t in TICKERS:
        passed, price = base_qmr_filter(t, date, indicators)
        if not passed:
            continue
        stock_ret5 = indicators[t]['ret5'].get(date)
        sector_etf = STOCK_SECTOR[t]
        sector_key = f'sector_{sector_etf}'
        if sector_key not in indicators:
            continue
        sector_ret5 = indicators[sector_key]['ret5'].get(date)
        if pd.isna(stock_ret5) or pd.isna(sector_ret5):
            continue
        if stock_ret5 < sector_ret5:
            signals.append((t, price))
    return signals


def generate_signals_E(date, indicators, close):
    """Variant E: QMR + VIX < 25 (calm market dips more likely to reverse)."""
    signals = []
    vix = indicators['VIX']
    if date not in vix.index:
        return signals
    vix_val = vix.get(date) if isinstance(vix, pd.Series) else None
    if vix_val is None or pd.isna(vix_val):
        return signals
    if float(vix_val) >= 25:
        return signals
    for t in TICKERS:
        passed, price = base_qmr_filter(t, date, indicators)
        if passed:
            signals.append((t, price))
    return signals


def generate_signals_F(date, indicators, close):
    """Variant F: QMR + sector ETF not in bottom 3 sectors by 1-month return."""
    # Rank all sectors by 20-day return
    sector_rets = {}
    for etf in SECTOR_ETFS:
        sector_key = f'sector_{etf}'
        if sector_key not in indicators:
            continue
        ret20 = indicators[sector_key]['ret20'].get(date)
        if pd.notna(ret20):
            sector_rets[etf] = float(ret20)

    if len(sector_rets) < 4:
        return []

    # Find bottom 3 sectors
    sorted_sectors = sorted(sector_rets.items(), key=lambda x: x[1])
    bottom3 = {s[0] for s in sorted_sectors[:3]}

    signals = []
    for t in TICKERS:
        passed, price = base_qmr_filter(t, date, indicators)
        if not passed:
            continue
        sector_etf = STOCK_SECTOR[t]
        if sector_etf in bottom3:
            continue  # Skip stocks in bottom 3 sectors
        signals.append((t, price))
    return signals


VARIANTS = {
    'A': {'signal_fn': generate_signals_A, 'desc': 'QMR baseline (no sector filter)'},
    'B': {'signal_fn': generate_signals_B, 'desc': 'QMR + sector RSI < 40'},
    'C': {'signal_fn': generate_signals_C, 'desc': 'QMR + sector 20d ret > 0 (recovering)'},
    'D': {'signal_fn': generate_signals_D, 'desc': 'QMR + stock lagging its sector (5d)'},
    'E': {'signal_fn': generate_signals_E, 'desc': 'QMR + VIX < 25 (calm market)'},
    'F': {'signal_fn': generate_signals_F, 'desc': 'QMR + sector not in bottom 3 (1mo ret)'},
}


def run_backtest(close, indicators, signal_fn):
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
                exit_price = float(exit_price_raw) * (1 - SLIPPAGE_PCT)
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
                    'sector': pos['sector'],
                })
            else:
                still_open.append(pos)
        positions = still_open

        # Generate signals and open new positions
        if len(positions) < MAX_POSITIONS:
            signals = signal_fn(date, indicators, close)
            held_tickers = {p['ticker'] for p in positions}
            signals = [(t, p) for t, p in signals if t not in held_tickers]

            slots = MAX_POSITIONS - len(positions)
            for t, price in signals[:slots]:
                entry_price = float(price) * (1 + SLIPPAGE_PCT)
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
                    'sector': STOCK_SECTOR.get(t, 'UNK'),
                })

        # Mark-to-market
        mtm = capital
        for pos in positions:
            curr = close.loc[date, pos['ticker']] if date in close.index else pos['entry_price']
            if pd.isna(curr):
                curr = pos['entry_price']
            mtm += float(curr) * pos['shares']
        equity[date] = mtm

    # Close remaining positions at last date
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
            'sector': pos.get('sector', 'UNK'),
        })

    equity_series = pd.Series(equity).sort_index()
    return trades, equity_series


def compute_metrics(trades, equity_series):
    """Compute performance metrics from trades and equity curve."""
    if len(trades) == 0:
        return {
            'n_trades': 0, 'total_pnl': 0, 'total_return_pct': 0,
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
            'max_drawdown_pct': 0, 'avg_trade_return_pct': 0,
        }

    total_pnl = sum(t['pnl'] for t in trades)
    total_return = (equity_series.iloc[-1] / STARTING_CAPITAL - 1) if len(equity_series) > 0 else 0

    # Daily returns from equity curve
    if len(equity_series) > 1:
        daily_rets = equity_series.pct_change().dropna()
        ann_factor = np.sqrt(252)
        sharpe = (daily_rets.mean() / daily_rets.std() * ann_factor) if daily_rets.std() > 0 else 0
        downside = daily_rets[daily_rets < 0].std()
        sortino = (daily_rets.mean() / downside * ann_factor) if downside > 0 else 0
    else:
        sharpe = 0
        sortino = 0

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

    trade_returns = [t['return'] for t in trades]
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
    close, vix_close = download_data()
    indicators = compute_indicators(close, vix_close)

    results = {}
    summary_rows = []
    baseline_metrics = None

    for variant_name, cfg in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Running Variant {variant_name}: {cfg['desc']}")
        print(f"{'='*60}")

        trades, equity = run_backtest(close, indicators, cfg['signal_fn'])
        metrics = compute_metrics(trades, equity)
        regime = compute_regime_sharpe(trades)
        perm_p = permutation_test(trades, equity)
        gates = five_gate_validation(metrics, regime, perm_p)

        if variant_name == 'A':
            baseline_metrics = metrics

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

        # Sector breakdown
        sector_pnl = defaultdict(lambda: {'pnl': 0, 'n': 0})
        for t in trades:
            s = t.get('sector', 'UNK')
            sector_pnl[s]['pnl'] += t['pnl']
            sector_pnl[s]['n'] += 1
        if sector_pnl:
            print(f"  Sector breakdown:")
            for s in sorted(sector_pnl.keys()):
                sp = sector_pnl[s]
                print(f"    {s}: {sp['n']} trades, ${sp['pnl']:.2f}")

        results[variant_name] = {
            'description': cfg['desc'],
            'hold_days': HOLD_DAYS,
            'metrics': metrics,
            'regime': regime,
            'permutation_p_value': perm_p,
            'gates': {k: bool(v) for k, v in gates.items()},
            'sector_breakdown': {s: {'pnl': round(v['pnl'], 2), 'n_trades': v['n']}
                                for s, v in sector_pnl.items()},
            'sample_trades': trades[:5] if len(trades) > 5 else trades,
        }

        summary_rows.append({
            'Variant': variant_name,
            'Desc': cfg['desc'][:40],
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
    print(f"\n{'='*120}")
    print("SECTOR MOMENTUM + QUALITY STOCK OVERLAY — SUMMARY TABLE")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL} | Slippage: {SLIPPAGE_PCT*100:.2f}%/side | Hold: {HOLD_DAYS}d")
    print(f"Base QMR condition: stock drops >5% from 20d high AND RSI(14) < 35")
    print(f"{'='*120}")

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    # ── Compare to baseline (Variant A) ──────────────────────────────────
    if baseline_metrics:
        print(f"\n{'='*80}")
        print("COMPARISON TO QMR BASELINE (Variant A)")
        print(f"{'='*80}")
        print(f"{'Variant':<10} {'Sharpe':>8} {'vs A':>8} {'PF':>8} {'vs A':>8} {'WR':>8} {'vs A':>8} {'Trades':>8}")
        print(f"{'-'*70}")
        for row in summary_rows:
            v = row['Variant']
            var_data = results[v]['metrics']
            sharpe_diff = var_data['sharpe'] - baseline_metrics['sharpe']
            pf_diff = var_data['profit_factor'] - baseline_metrics['profit_factor']
            wr_diff = var_data['win_rate'] - baseline_metrics['win_rate']
            prefix = '' if v == 'A' else ('+' if sharpe_diff >= 0 else '')
            print(f"{v:<10} {var_data['sharpe']:>8.3f} {prefix}{sharpe_diff:>7.3f} "
                  f"{var_data['profit_factor']:>8.2f} {'+' if pf_diff >= 0 else ''}{pf_diff:>7.2f} "
                  f"{var_data['win_rate']:>7.1%} {'+' if wr_diff >= 0 else ''}{wr_diff:>7.1%} "
                  f"{var_data['n_trades']:>8}")

    passed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'PASS']
    failed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'FAIL']
    print(f"\nPASSED 5-GATE: {passed_variants if passed_variants else 'None'}")
    print(f"FAILED 5-GATE: {failed_variants if failed_variants else 'None'}")

    # Key finding
    if baseline_metrics:
        improving = [(r['Variant'], results[r['Variant']]['metrics']['sharpe'])
                     for r in summary_rows
                     if r['Variant'] != 'A' and results[r['Variant']]['metrics']['sharpe'] > baseline_metrics['sharpe']]
        if improving:
            best = max(improving, key=lambda x: x[1])
            print(f"\nBEST IMPROVEMENT: Variant {best[0]} (Sharpe {best[1]:.3f} vs baseline {baseline_metrics['sharpe']:.3f})")
        else:
            print(f"\nNO VARIANT IMPROVED on baseline QMR Sharpe ({baseline_metrics['sharpe']:.3f})")

    # ── Save results ───────────────────────────────────────────────────────
    results['_meta'] = {
        'strategy': 'Sector Momentum + Quality Stock Overlay',
        'universe': TICKERS,
        'sector_map': STOCK_SECTOR,
        'sector_etfs': SECTOR_ETFS,
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'starting_capital': STARTING_CAPITAL,
        'slippage_pct_each_way': SLIPPAGE_PCT,
        'hold_days': HOLD_DAYS,
        'max_positions': MAX_POSITIONS,
        'max_per_trade': MAX_PER_TRADE,
        'n_permutations': N_PERMUTATIONS,
        'run_timestamp': datetime.now().isoformat(),
        'passed_variants': passed_variants,
        'failed_variants': failed_variants,
        'base_qmr_condition': 'stock drops >5% from 20-day high AND RSI(14) < 35',
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
