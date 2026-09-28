#!/usr/bin/env python3
"""
Small-Cap Quality Mean Reversion Backtest
Tests whether the validated QMR strategy works on small/mid-cap quality stocks.
Thesis: higher volatility = more dips = more opportunities, less analyst coverage = more mispricing.
6 variants, 5-gate validation.
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
    'DECK', 'POOL', 'WSM', 'TTC', 'LSTR', 'CASY', 'WING', 'TXRH', 'DINO', 'FIX',
    'BWXT', 'SAIA', 'EXPO', 'CSWI', 'TREX', 'RBC', 'ENSG', 'CWST', 'MGEE', 'CALM'
]
SPY = 'SPY'
VIX_TICKER = '^VIX'
START_DATE = '2021-01-01'  # extra lookback for indicators
OOT_START = '2022-01-01'
OOT_END = '2026-07-31'
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0005  # 0.05% each way (higher for small caps)
MAX_POSITIONS = 3
MAX_PER_TRADE = 200.0
N_PERMUTATIONS = 1000

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/smallcap_quality_mr_results.json'


def download_data():
    """Download all price and volume data."""
    all_tickers = TICKERS + [SPY]
    print(f"Downloading price data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns
    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    close = close.ffill().dropna(how='all')

    volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else None
    if volume is not None:
        volume = volume.ffill().fillna(0)

    # Download VIX separately
    print("Downloading VIX data...")
    vix_data = yf.download(VIX_TICKER, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    vix_close = vix_data['Close'].squeeze() if len(vix_data) > 0 else pd.Series(dtype=float)

    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, volume, vix_close


def compute_rsi(series, period=14):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_indicators(close, volume, vix_close):
    """Pre-compute all indicators needed for signals."""
    indicators = {}

    for t in TICKERS:
        if t not in close.columns:
            print(f"  WARNING: {t} not in data, skipping")
            continue
        s = close[t]
        ind = {
            'close': s,
            'rsi14': compute_rsi(s, 14),
            'high20': s.rolling(20).max(),
        }
        # Volume indicators (for variant E)
        if volume is not None and t in volume.columns:
            vol_series = volume[t]
            ind['volume'] = vol_series
            ind['vol_sma20'] = vol_series.rolling(20).mean()

        indicators[t] = ind

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


def _get_vals(date, indicators, ticker, keys):
    """Helper to retrieve multiple indicator values for a ticker on a date."""
    if ticker not in indicators:
        return None
    ind = indicators[ticker]
    if date not in ind['close'].index:
        return None
    vals = {}
    for k in keys:
        if k not in ind:
            return None
        v = ind[k].get(date) if hasattr(ind[k], 'get') else None
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        vals[k] = v
    return vals


# ── Signal Generators ──────────────────────────────────────────────────────

def generate_signals_A(date, indicators, close):
    """Variant A: Standard QMR-A on small-cap quality universe.
    Stock drops >5% from 20-day high AND RSI(14) < 35. Hold 10 days."""
    signals = []
    for t in TICKERS:
        vals = _get_vals(date, indicators, t, ['close', 'high20', 'rsi14'])
        if vals is None:
            continue
        drawdown = (vals['close'] - vals['high20']) / vals['high20']
        if drawdown < -0.05 and vals['rsi14'] < 35:
            signals.append((t, float(vals['close'])))
    return signals


def generate_signals_B(date, indicators, close):
    """Variant B: QMR-A with 7% drawdown threshold (deeper dips on volatile stocks).
    Stock drops >7% from 20-day high AND RSI(14) < 35. Hold 10 days."""
    signals = []
    for t in TICKERS:
        vals = _get_vals(date, indicators, t, ['close', 'high20', 'rsi14'])
        if vals is None:
            continue
        drawdown = (vals['close'] - vals['high20']) / vals['high20']
        if drawdown < -0.07 and vals['rsi14'] < 35:
            signals.append((t, float(vals['close'])))
    return signals


def generate_signals_C(date, indicators, close):
    """Variant C: QMR-A with RSI < 25 (more extreme oversold).
    Stock drops >5% from 20-day high AND RSI(14) < 25. Hold 10 days."""
    signals = []
    for t in TICKERS:
        vals = _get_vals(date, indicators, t, ['close', 'high20', 'rsi14'])
        if vals is None:
            continue
        drawdown = (vals['close'] - vals['high20']) / vals['high20']
        if drawdown < -0.05 and vals['rsi14'] < 25:
            signals.append((t, float(vals['close'])))
    return signals


def generate_signals_D(date, indicators, close):
    """Variant D: QMR-A but hold 15 days (small caps may need longer to recover).
    Stock drops >5% from 20-day high AND RSI(14) < 35. Hold 15 days.
    (Signal identical to A; hold period differs in VARIANTS config.)"""
    return generate_signals_A(date, indicators, close)


def generate_signals_E(date, indicators, close):
    """Variant E: QMR-A + volume spike (daily vol > 1.5x 20-day avg — institutional selling).
    Stock drops >5% from 20-day high AND RSI(14) < 35 AND volume spike. Hold 10 days."""
    signals = []
    for t in TICKERS:
        vals = _get_vals(date, indicators, t, ['close', 'high20', 'rsi14'])
        if vals is None:
            continue
        # Check volume spike
        ind = indicators[t]
        if 'volume' not in ind or 'vol_sma20' not in ind:
            continue
        vol_today = ind['volume'].get(date)
        vol_avg = ind['vol_sma20'].get(date)
        if vol_today is None or vol_avg is None or pd.isna(vol_today) or pd.isna(vol_avg) or vol_avg == 0:
            continue

        drawdown = (vals['close'] - vals['high20']) / vals['high20']
        if drawdown < -0.05 and vals['rsi14'] < 35 and vol_today > 1.5 * vol_avg:
            signals.append((t, float(vals['close'])))
    return signals


def generate_signals_F(date, indicators, close):
    """Variant F: QMR-A + VIX < 25 filter.
    Stock drops >5% from 20-day high AND RSI(14) < 35 AND VIX < 25. Hold 10 days."""
    # Check VIX first
    vix = indicators.get('VIX')
    if vix is None or date not in vix.index:
        return []
    vix_val = vix.get(date) if hasattr(vix, 'get') else None
    if vix_val is None or (isinstance(vix_val, float) and pd.isna(vix_val)):
        return []
    if float(vix_val) >= 25:
        return []

    # Same as A but only when VIX < 25
    return generate_signals_A(date, indicators, close)


VARIANTS = {
    'A': {'signal_fn': generate_signals_A, 'hold_days': 10, 'desc': 'Standard QMR-A: 5% DD + RSI<35, hold 10d'},
    'B': {'signal_fn': generate_signals_B, 'hold_days': 10, 'desc': '7% DD threshold + RSI<35, hold 10d'},
    'C': {'signal_fn': generate_signals_C, 'hold_days': 10, 'desc': '5% DD + RSI<25 (extreme oversold), hold 10d'},
    'D': {'signal_fn': generate_signals_D, 'hold_days': 15, 'desc': 'Standard QMR-A, hold 15d (longer recovery)'},
    'E': {'signal_fn': generate_signals_E, 'hold_days': 10, 'desc': '5% DD + RSI<35 + volume spike >1.5x, hold 10d'},
    'F': {'signal_fn': generate_signals_F, 'hold_days': 10, 'desc': '5% DD + RSI<35 + VIX<25 filter, hold 10d'},
}


def run_backtest(close, indicators, signal_fn, hold_days):
    """Run a single variant backtest. Returns list of trades and daily equity curve."""
    oot_dates = close.loc[OOT_START:OOT_END].index
    if len(oot_dates) == 0:
        return [], pd.Series(dtype=float)

    capital = STARTING_CAPITAL
    positions = []  # list of {ticker, entry_price, entry_date, shares, exit_target}
    trades = []     # completed trades
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
                exit_price = float(exit_price_raw) * (1 - SLIPPAGE_PCT)  # sell slippage
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

        # Generate signals and enter new positions
        if len(positions) < MAX_POSITIONS:
            signals = signal_fn(date, indicators, close)
            held_tickers = {p['ticker'] for p in positions}
            signals = [(t, p) for t, p in signals if t not in held_tickers]

            slots = MAX_POSITIONS - len(positions)
            for t, price in signals[:slots]:
                entry_price = price * (1 + SLIPPAGE_PCT)  # buy slippage
                alloc = min(MAX_PER_TRADE, capital * 0.95)
                if alloc < 10 or capital < 10:
                    continue
                shares = int(alloc / entry_price)
                if shares < 1:
                    continue
                cost = shares * entry_price
                capital -= cost
                regime = get_regime(date, indicators)
                idx_loc = oot_dates.get_loc(date)
                exit_target = oot_dates[min(idx_loc + hold_days, len(oot_dates) - 1)]
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

    # Close any remaining positions at last date
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
    """Compute performance metrics from trades and equity curve."""
    if len(trades) == 0:
        return {
            'n_trades': 0, 'total_pnl': 0, 'total_return_pct': 0,
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
            'max_drawdown_pct': 0, 'avg_trade_return_pct': 0,
        }

    trade_returns = [t['return'] for t in trades]
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

    # Profit factor
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win rate
    winners = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winners / len(trades)

    # Max drawdown
    if len(equity_series) > 0:
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
    else:
        max_dd = 0

    avg_trade_ret = np.mean(trade_returns) * 100

    return {
        'n_trades': len(trades),
        'total_pnl': round(float(total_pnl), 2),
        'total_return_pct': round(float(total_return * 100), 2),
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(profit_factor), 3),
        'win_rate': round(float(win_rate), 3),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
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
    """Sign-shuffle permutation test on trade returns. Returns p-value."""
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
    close, volume, vix_close = download_data()
    indicators = compute_indicators(close, volume, vix_close)

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

        # Per-ticker breakdown
        ticker_pnl = defaultdict(float)
        ticker_count = defaultdict(int)
        for t in trades:
            ticker_pnl[t['ticker']] += t['pnl']
            ticker_count[t['ticker']] += 1
        top_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
        if top_tickers:
            print(f"  Top tickers: {', '.join(f'{t}=${p:.0f}({ticker_count[t]})' for t, p in top_tickers[:5])}")

        results[variant_name] = {
            'description': cfg['desc'],
            'hold_days': cfg['hold_days'],
            'metrics': metrics,
            'regime': regime,
            'permutation_p_value': perm_p,
            'gates': {k: bool(v) for k, v in gates.items()},
            'ticker_breakdown': {t: {'pnl': round(p, 2), 'trades': ticker_count[t]} for t, p in top_tickers},
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
    print("SMALL-CAP QUALITY MEAN REVERSION — SUMMARY TABLE")
    print(f"Universe: {len(TICKERS)} small/mid-cap quality stocks")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL} | Slippage: {SLIPPAGE_PCT*100:.2f}%/side")
    print(f"{'='*100}")

    df = pd.DataFrame(summary_rows)
    print(df.to_string(index=False))

    passed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'PASS']
    failed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'FAIL']
    print(f"\nPASSED 5-GATE: {passed_variants if passed_variants else 'None'}")
    print(f"FAILED 5-GATE: {failed_variants if failed_variants else 'None'}")

    # ── Comparison note ────────────────────────────────────────────────────
    print(f"\n--- Small-Cap vs Large-Cap QMR Thesis ---")
    print(f"Small caps tested: {', '.join(TICKERS)}")
    print(f"Higher slippage (0.05% vs 0.02%) accounts for wider spreads in small caps.")
    print(f"If small-cap QMR passes, it suggests mispricing is broader than mega-cap quality names.")

    # ── Save results ───────────────────────────────────────────────────────
    results['_meta'] = {
        'strategy': 'Small-Cap Quality Mean Reversion',
        'universe': TICKERS,
        'universe_description': 'Well-known profitable small/mid caps with stable businesses',
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
