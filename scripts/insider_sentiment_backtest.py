#!/usr/bin/env python3
"""
Insider/Institutional Sentiment Backtest on Quality Stocks
Variants A-F using proxy signals for insider/institutional activity.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Parameters ──────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP', 'HD', 'COST',
    'UNH', 'LLY', 'V', 'MA', 'ABBV', 'MRK', 'WMT', 'AMZN', 'GOOGL', 'META'
]
START = '2022-01-01'
END = '2026-07-31'
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 2
N_PERMUTATIONS = 1000

# ── Data Download ───────────────────────────────────────────────────────
print("Downloading data...")
data = {}
for ticker in UNIVERSE:
    try:
        df = yf.download(ticker, start=START, end=END, progress=False)
        if len(df) > 30:
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            data[ticker] = df
            print(f"  {ticker}: {len(df)} bars")
        else:
            print(f"  {ticker}: insufficient data, skipping")
    except Exception as e:
        print(f"  {ticker}: download failed: {e}")

print(f"Loaded {len(data)} tickers")

# ── Feature Computation ─────────────────────────────────────────────────
def compute_features(df):
    """Compute all technical features needed for variants A-F."""
    df = df.copy()
    df['ret'] = df['Close'].pct_change()
    df['vol_ma20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / df['vol_ma20']
    df['high20'] = df['Close'].rolling(20).max()
    df['pct_from_high20'] = (df['Close'] - df['high20']) / df['high20']
    df['green'] = df['Close'] > df['Open']

    # 5-day return
    df['ret_5d'] = df['Close'].pct_change(5)

    # OBV
    obv = [0.0]
    closes = df['Close'].values
    volumes = df['Volume'].values
    for i in range(1, len(df)):
        if closes[i] > closes[i-1]:
            obv.append(obv[-1] + volumes[i])
        elif closes[i] < closes[i-1]:
            obv.append(obv[-1] - volumes[i])
        else:
            obv.append(obv[-1])
    df['obv'] = obv
    df['obv_high20'] = df['obv'].rolling(20).max()

    # CMF (20-period)
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.fillna(0)
    mfv = mfm * df['Volume']
    df['cmf'] = mfv.rolling(20).sum() / df['Volume'].rolling(20).sum()

    # RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))

    return df

print("Computing features...")
for ticker in data:
    data[ticker] = compute_features(data[ticker])

# ── Signal Generation ───────────────────────────────────────────────────
def generate_signals(data, variant):
    """Generate buy signals for a given variant. Returns list of (date, ticker)."""
    signals = []
    for ticker, df in data.items():
        for i in range(25, len(df)):
            row = df.iloc[i]
            date = df.index[i]

            if variant == 'A':
                # Low-volume dip: drop >3%, volume < 0.7x avg
                if row['ret'] < -0.03 and row['vol_ratio'] < 0.7:
                    signals.append((date, ticker))

            elif variant == 'B':
                # High volume recovery after >5% drop in last 5 days
                if row['ret_5d'] < -0.05 and row['green'] and row['vol_ratio'] > 1.5:
                    signals.append((date, ticker))

            elif variant == 'C':
                # OBV new 20-day high while price >5% below 20-day high
                if row['pct_from_high20'] < -0.05 and row['obv'] >= row['obv_high20'] - 1e-6:
                    signals.append((date, ticker))

            elif variant == 'D':
                # Down >5% from 20d high, CMF positive
                if row['pct_from_high20'] < -0.05 and row['cmf'] > 0:
                    signals.append((date, ticker))

            elif variant == 'E':
                # Drop >5% from high, RSI < 40, volume < 0.8x avg
                if row['pct_from_high20'] < -0.05 and row['rsi'] < 40 and row['vol_ratio'] < 0.8:
                    signals.append((date, ticker))

            elif variant == 'F':
                # Combined: at least 2 of (OBV div, CMF positive, low-vol selloff)
                # while down >5% from 20d high
                if row['pct_from_high20'] < -0.05:
                    score = 0
                    # OBV divergence (C condition)
                    if row['obv'] >= row['obv_high20'] - 1e-6:
                        score += 1
                    # CMF positive (D condition)
                    if row['cmf'] > 0:
                        score += 1
                    # Low-volume selloff (A-like: vol < 0.7x)
                    if row['vol_ratio'] < 0.7:
                        score += 1
                    if score >= 2:
                        signals.append((date, ticker))

    return signals

# ── Backtest Engine ─────────────────────────────────────────────────────
def run_backtest(data, signals, capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
                 max_concurrent=MAX_CONCURRENT, hold_days=HOLD_DAYS,
                 slippage_bps=SLIPPAGE_BPS):
    """Run backtest with position sizing, concurrency limits, and slippage."""
    trades = []
    active_positions = []  # list of (exit_date_idx, ticker, shares, entry_price, entry_date)

    # Build date index mapping per ticker
    date_indices = {}
    for ticker, df in data.items():
        date_indices[ticker] = {d: i for i, d in enumerate(df.index)}

    for sig_date, ticker in sorted(signals, key=lambda x: x[0]):
        # Remove expired positions
        df = data[ticker]
        if ticker not in date_indices or sig_date not in date_indices[ticker]:
            continue

        sig_idx = date_indices[ticker][sig_date]

        # Count currently active positions at this date
        active_count = sum(1 for pos in active_positions
                          if pos[4] <= sig_date and
                          (pos[0] is None or pos[0] > sig_date))

        if active_count >= max_concurrent:
            continue

        # Entry: next day open (or close of signal day as proxy)
        entry_idx = sig_idx + 1
        if entry_idx >= len(df):
            continue

        entry_price = df.iloc[entry_idx]['Open']
        if pd.isna(entry_price) or entry_price <= 0:
            continue

        # Apply slippage on entry
        entry_price *= (1 + slippage_bps / 10000)

        # Position sizing
        shares = int(max_per_trade / entry_price)
        if shares < 1:
            continue

        # Exit: hold_days later
        exit_idx = entry_idx + hold_days
        if exit_idx >= len(df):
            exit_idx = len(df) - 1

        exit_price = df.iloc[exit_idx]['Close']
        if pd.isna(exit_price) or exit_price <= 0:
            continue

        # Apply slippage on exit
        exit_price *= (1 - slippage_bps / 10000)

        entry_date = df.index[entry_idx]
        exit_date = df.index[exit_idx]

        pnl = (exit_price - entry_price) * shares
        ret = (exit_price / entry_price) - 1

        trades.append({
            'ticker': ticker,
            'entry_date': str(entry_date.date()),
            'exit_date': str(exit_date.date()),
            'entry_price': round(float(entry_price), 2),
            'exit_price': round(float(exit_price), 2),
            'shares': shares,
            'pnl': round(float(pnl), 2),
            'return': round(float(ret), 6),
        })

        active_positions.append((exit_date, ticker, shares, entry_price, entry_date))

    return trades

# ── Metrics ─────────────────────────────────────────────────────────────
def compute_metrics(trades, capital=CAPITAL):
    """Compute strategy metrics from trade list."""
    if not trades:
        return {
            'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'profit_factor': 0,
            'win_rate': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0,
            'avg_return_pct': 0, 'median_return_pct': 0,
        }

    returns = np.array([t['return'] for t in trades])
    pnls = np.array([t['pnl'] for t in trades])

    n_trades = len(trades)
    wins = returns > 0
    win_rate = wins.sum() / n_trades

    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    # Annualized Sharpe (assume ~25 trades/year roughly, use daily-like scaling)
    # Use per-trade returns, annualize assuming ~252/hold_days trades per year
    trades_per_year = 252 / HOLD_DAYS
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Equity curve and max drawdown
    equity = [capital]
    for pnl in pnls:
        equity.append(equity[-1] + pnl)
    equity = np.array(equity)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    total_return_pct = (equity[-1] / capital - 1) * 100

    return {
        'n_trades': n_trades,
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'profit_factor': round(float(profit_factor), 3),
        'win_rate': round(float(win_rate), 4),
        'total_return_pct': round(float(total_return_pct), 2),
        'max_drawdown_pct': round(float(max_dd * 100), 2),
        'avg_return_pct': round(float(returns.mean() * 100), 4),
        'median_return_pct': round(float(np.median(returns) * 100), 4),
        'total_pnl': round(float(pnls.sum()), 2),
    }

# ── Regime Analysis ─────────────────────────────────────────────────────
def regime_analysis(trades, spy_data):
    """Split trades into bull/bear regimes using SPY 50-day MA."""
    if not trades or spy_data is None:
        return {'bull_sharpe': 0, 'bear_sharpe': 0, 'regime_gap': 1.0}

    spy = spy_data.copy()
    spy['ma50'] = spy['Close'].rolling(50).mean()
    spy['regime'] = np.where(spy['Close'] > spy['ma50'], 'bull', 'bear')

    bull_rets, bear_rets = [], []
    for t in trades:
        entry = pd.Timestamp(t['entry_date'])
        # Find closest date in SPY
        mask = spy.index <= entry
        if not mask.any():
            continue
        closest = spy.index[mask][-1]
        regime = spy.loc[closest, 'regime']
        if regime == 'bull':
            bull_rets.append(t['return'])
        else:
            bear_rets.append(t['return'])

    trades_per_year = 252 / HOLD_DAYS

    def regime_sharpe(rets):
        if len(rets) < 3:
            return 0
        r = np.array(rets)
        return (r.mean() / r.std()) * np.sqrt(trades_per_year) if r.std() > 0 else 0

    bull_s = regime_sharpe(bull_rets)
    bear_s = regime_sharpe(bear_rets)
    denom = max(abs(bull_s), abs(bear_s), 1e-9)
    gap = abs(bull_s - bear_s) / denom

    return {
        'bull_sharpe': round(float(bull_s), 3),
        'bear_sharpe': round(float(bear_s), 3),
        'regime_gap': round(float(gap), 3),
        'bull_trades': len(bull_rets),
        'bear_trades': len(bear_rets),
    }

# ── Permutation Test ────────────────────────────────────────────────────
def permutation_test(data, signals, real_sharpe, n_perms=N_PERMUTATIONS):
    """Shuffle signal dates randomly to test if edge is real."""
    if not signals or real_sharpe == 0:
        return 1.0

    # Get all valid trading dates across all tickers
    all_dates = set()
    for ticker, df in data.items():
        all_dates.update(df.index[25:].tolist())
    all_dates = sorted(all_dates)

    tickers = list(data.keys())
    n_signals = len(signals)
    count_better = 0

    for _ in range(n_perms):
        # Random signals: pick random (date, ticker) pairs
        rand_dates = np.random.choice(len(all_dates), size=n_signals, replace=True)
        rand_tickers = np.random.choice(tickers, size=n_signals, replace=True)
        rand_signals = [(all_dates[d], t) for d, t in zip(rand_dates, rand_tickers)]

        rand_trades = run_backtest(data, rand_signals)
        if len(rand_trades) < 5:
            continue
        rand_metrics = compute_metrics(rand_trades)
        if rand_metrics['sharpe'] >= real_sharpe:
            count_better += 1

    p_value = count_better / n_perms
    return round(p_value, 4)

# ── 5-Gate Validation ───────────────────────────────────────────────────
def validate_5gate(metrics, regime, p_value):
    """Check all 5 gates."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'permutation_p_lt_0.05': p_value < 0.05,
        'regime_gap_lt_0.5': regime['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'min_20_trades': metrics['n_trades'] >= 20,
    }
    gates['all_passed'] = all(gates.values())
    gates['gates_passed'] = sum(v for k, v in gates.items() if k != 'all_passed' and k != 'gates_passed')
    return gates

# ── Main ────────────────────────────────────────────────────────────────
def main():
    # Download SPY for regime analysis
    print("\nDownloading SPY for regime analysis...")
    spy_raw = yf.download('SPY', start=START, end=END, progress=False)
    spy_raw.columns = [c[0] if isinstance(c, tuple) else c for c in spy_raw.columns]

    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    results = {}

    for v in variants:
        print(f"\n{'='*60}")
        print(f"VARIANT {v}")
        print(f"{'='*60}")

        signals = generate_signals(data, v)
        print(f"  Signals generated: {len(signals)}")

        trades = run_backtest(data, signals)
        print(f"  Trades executed: {len(trades)}")

        metrics = compute_metrics(trades)
        print(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}")
        print(f"  PF: {metrics['profit_factor']}, WR: {metrics['win_rate']:.1%}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}%, Max DD: {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Total P&L: ${metrics['total_pnl']:.2f}")

        regime = regime_analysis(trades, spy_raw)
        print(f"  Bull Sharpe: {regime['bull_sharpe']}, Bear Sharpe: {regime['bear_sharpe']}")
        print(f"  Regime Gap: {regime['regime_gap']}")

        print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...")
        p_value = permutation_test(data, signals, metrics['sharpe'])
        print(f"  P-value: {p_value}")

        gates = validate_5gate(metrics, regime, p_value)
        print(f"  Gates passed: {gates['gates_passed']}/5 — {'PASS' if gates['all_passed'] else 'FAIL'}")

        results[f'variant_{v}'] = {
            'description': {
                'A': 'Low-volume dip buy (>3% drop, vol < 0.7x avg)',
                'B': 'High-volume recovery (>5% drop in 5d, green day vol > 1.5x)',
                'C': 'OBV divergence (OBV new 20d high, price >5% below 20d high)',
                'D': 'CMF divergence (price >5% below 20d high, CMF > 0)',
                'E': 'Institutional dip buy (>5% drop, RSI < 40, vol < 0.8x)',
                'F': 'Combined accumulation (2+ of C, D, A conditions, >5% below high)',
            }[v],
            'signals': len(signals),
            'metrics': metrics,
            'regime': regime,
            'p_value': p_value,
            'gates': gates,
            'sample_trades': trades[:5] if trades else [],
        }

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Var':<4} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'Return%':>8} {'MaxDD%':>7} {'P-val':>6} {'Gates':>5}")
    print("-" * 75)

    best_variant = None
    best_sharpe = -999

    for v in variants:
        r = results[f'variant_{v}']
        m = r['metrics']
        g = r['gates']
        status = 'PASS' if g['all_passed'] else 'FAIL'
        print(f"  {v:<3} {m['n_trades']:>6} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.2f} {m['win_rate']:>5.1%} {m['total_return_pct']:>7.1f}% {m['max_drawdown_pct']:>6.1f}% {r['p_value']:>6.4f} {g['gates_passed']}/5 {status}")

        if m['sharpe'] > best_sharpe and m['n_trades'] >= 20:
            best_sharpe = m['sharpe']
            best_variant = v

    results['best_variant'] = best_variant
    results['metadata'] = {
        'strategy': 'Insider/Institutional Sentiment on Quality Stocks',
        'universe': UNIVERSE,
        'period': f'{START} to {END}',
        'capital': CAPITAL,
        'max_per_trade': MAX_PER_TRADE,
        'max_concurrent': MAX_CONCURRENT,
        'hold_days': HOLD_DAYS,
        'slippage_bps': SLIPPAGE_BPS,
        'n_permutations': N_PERMUTATIONS,
        'run_date': datetime.now().isoformat(),
    }

    # Save results
    output_path = '/home/jupiter/Lvl3Quant/data/insider_sentiment_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main()
