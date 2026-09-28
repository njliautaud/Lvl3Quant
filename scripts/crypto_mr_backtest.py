#!/usr/bin/env python3
"""
Crypto Mean Reversion Backtest — Dual Signal D Logic
Tests dip-buy + recovery confirmation on large-cap crypto proxies.
"""

import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────────
START = '2022-01-01'
END = '2026-07-31'
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 5  # 5 bps
N_PERMUTATIONS = 1000

VARIANTS = {
    'A': {
        'name': 'BTC proxy (BITO/GBTC)',
        'tickers': ['BITO', 'GBTC'],
        'dip_pct': 5, 'rsi_thresh': 35, 'red_days': 3,
    },
    'B': {
        'name': 'ETH proxy (ETHE)',
        'tickers': ['ETHE'],
        'dip_pct': 5, 'rsi_thresh': 35, 'red_days': 3,
    },
    'C': {
        'name': 'Broad crypto (BITO/ETHE/BITQ)',
        'tickers': ['BITO', 'ETHE', 'BITQ'],
        'dip_pct': 5, 'rsi_thresh': 35, 'red_days': 3,
    },
    'D': {
        'name': 'BTC relaxed thresholds',
        'tickers': ['BITO', 'GBTC'],
        'dip_pct': 10, 'rsi_thresh': 30, 'red_days': 2,
    },
    'E': {
        'name': 'BTC+ETH relaxed thresholds',
        'tickers': ['BITO', 'GBTC', 'ETHE'],
        'dip_pct': 10, 'rsi_thresh': 30, 'red_days': 2,
    },
    'F': {
        'name': 'Crypto-adjacent (COIN/MARA/MSTR)',
        'tickers': ['COIN', 'MARA', 'MSTR'],
        'dip_pct': 5, 'rsi_thresh': 35, 'red_days': 3,
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    """Standard RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def download_data(tickers):
    """Download daily OHLCV for all tickers + SPY for regime."""
    all_tickers = list(set(tickers + ['SPY']))
    print(f"  Downloading: {all_tickers}")
    data = {}
    for t in all_tickers:
        try:
            df = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df
                print(f"    {t}: {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
            else:
                print(f"    {t}: insufficient data ({0 if df is None else len(df)} bars), skipping")
        except Exception as e:
            print(f"    {t}: download failed ({e}), skipping")
    return data


def find_entries(df, dip_pct, rsi_thresh, red_days_req):
    """
    Dual Signal D entry logic:
    1. Price drops > dip_pct% from 20-day high
    2. RSI(14) < rsi_thresh
    3. First green day after red_days_req+ consecutive red days
    """
    close = df['Close'].values
    high20 = df['Close'].rolling(20).max().values
    rsi = compute_rsi(df['Close']).values

    # Green/red day classification
    is_green = np.array([False] + [close[i] > close[i-1] for i in range(1, len(close))])
    is_red = np.array([False] + [close[i] < close[i-1] for i in range(1, len(close))])

    # Count consecutive red days before each bar
    consec_red = np.zeros(len(close), dtype=int)
    for i in range(1, len(close)):
        if is_red[i-1]:
            consec_red[i] = consec_red[i-1] + 1
        else:
            consec_red[i] = 0

    entries = []
    for i in range(21, len(close)):
        if np.isnan(rsi[i]) or np.isnan(high20[i]):
            continue
        dip = (high20[i] - close[i]) / high20[i] * 100.0
        cond_dip = dip >= dip_pct
        cond_rsi = rsi[i] < rsi_thresh
        cond_recovery = is_green[i] and consec_red[i] >= red_days_req
        if cond_dip and cond_rsi and cond_recovery:
            entries.append(i)
    return entries


def run_backtest_single(df, entries, capital, max_per_trade, max_concurrent, hold_days, slippage_bps):
    """Simulate trades with position limits."""
    close = df['Close'].values
    dates = df.index
    trades = []
    active = []  # list of (entry_idx, entry_price, shares)

    for entry_idx in entries:
        # Remove expired positions
        active = [(ei, ep, sh) for ei, ep, sh in active if entry_idx - ei < hold_days]

        if len(active) >= max_concurrent:
            continue

        entry_price = close[entry_idx] * (1 + slippage_bps / 10000.0)
        shares = min(max_per_trade, capital / max(max_concurrent, 1)) / entry_price
        if shares < 0.001:
            continue

        exit_idx = min(entry_idx + hold_days, len(close) - 1)
        exit_price = close[exit_idx] * (1 - slippage_bps / 10000.0)

        pnl = (exit_price - entry_price) * shares
        ret = (exit_price / entry_price) - 1.0

        trades.append({
            'entry_date': str(dates[entry_idx].date()),
            'exit_date': str(dates[exit_idx].date()),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'shares': float(shares),
            'pnl': float(pnl),
            'return': float(ret),
        })
        active.append((entry_idx, entry_price, shares))

    return trades


def compute_metrics(trades, capital):
    """Compute performance metrics from trade list."""
    if not trades:
        return {
            'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
            'max_drawdown': 0, 'total_return': 0, 'num_trades': 0,
        }

    returns = np.array([t['return'] for t in trades])
    pnls = np.array([t['pnl'] for t in trades])

    # Annualized (assume ~25 trades/year baseline, scale by actual)
    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9

    # Annualize assuming each trade is ~10 days, ~252/10 = ~25 trades per year
    trades_per_year = 252.0 / HOLD_DAYS
    ann_factor = np.sqrt(trades_per_year)

    sharpe = (mean_ret / std_ret) * ann_factor if std_ret > 1e-9 else 0
    sortino = (mean_ret / downside_std) * ann_factor if downside_std > 1e-9 else 0

    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    win_rate = len(wins) / n if n > 0 else 0
    gross_profit = np.sum(pnls[pnls > 0]) if len(pnls[pnls > 0]) > 0 else 0
    gross_loss = abs(np.sum(pnls[pnls < 0])) if len(pnls[pnls < 0]) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-9 else float('inf')

    # Equity curve for drawdown
    equity = capital + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdowns = (equity - peak) / peak
    max_dd = float(np.min(drawdowns))

    total_ret = float(np.sum(pnls) / capital)

    return {
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'win_rate': round(float(win_rate), 4),
        'profit_factor': round(float(profit_factor), 4),
        'max_drawdown': round(float(max_dd), 4),
        'total_return': round(float(total_ret), 4),
        'num_trades': int(n),
    }


def regime_split(trades, spy_df):
    """Split trades into bull/bear based on SPY > 200-SMA."""
    if not trades or spy_df is None or len(spy_df) < 200:
        return [], []

    spy_close = spy_df['Close']
    spy_sma200 = spy_close.rolling(200).mean()

    bull_trades, bear_trades = [], []
    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        # Find nearest SPY date
        mask = spy_df.index <= entry_date
        if not mask.any():
            continue
        idx = spy_df.index[mask][-1]
        if pd.isna(spy_sma200.loc[idx]):
            continue
        if spy_close.loc[idx] > spy_sma200.loc[idx]:
            bull_trades.append(t)
        else:
            bear_trades.append(t)

    return bull_trades, bear_trades


def permutation_test(trades, all_dates_count, capital, n_perm=1000):
    """
    Permutation test: shuffle entry dates randomly among tradeable days,
    measure how often random achieves >= actual Sharpe.
    """
    if len(trades) < 5:
        return 1.0  # not enough trades

    actual_sharpe = compute_metrics(trades, capital)['sharpe']
    n_trades = len(trades)

    count_better = 0
    returns_pool = np.array([t['return'] for t in trades])

    for _ in range(n_perm):
        # Shuffle the returns to simulate random entry timing
        shuffled = np.random.permutation(returns_pool)
        mean_r = np.mean(shuffled)
        std_r = np.std(shuffled, ddof=1) if len(shuffled) > 1 else 1e-9
        ann_factor = np.sqrt(252.0 / HOLD_DAYS)
        rand_sharpe = (mean_r / std_r) * ann_factor if std_r > 1e-9 else 0
        if rand_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perm


def five_gate_check(metrics, regime_gap, perm_p):
    """Apply the 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'max_dd_gt_neg50': metrics['max_drawdown'] > -0.50,
        'min_20_trades': metrics['num_trades'] >= 20,
    }
    gates['all_pass'] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("CRYPTO MEAN REVERSION BACKTEST — Dual Signal D Logic")
    print(f"Period: {START} to {END} | Capital: ${CAPITAL}")
    print("=" * 70)

    # Download all unique tickers across variants + SPY
    all_tickers = set(['SPY'])
    for v in VARIANTS.values():
        all_tickers.update(v['tickers'])
    all_tickers = sorted(all_tickers)

    print("\n[1] Downloading data...")
    data = download_data(all_tickers)
    spy_df = data.get('SPY')

    results = {}

    for var_key, var_cfg in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Variant {var_key}: {var_cfg['name']}")
        print(f"  Dip: {var_cfg['dip_pct']}% | RSI < {var_cfg['rsi_thresh']} | {var_cfg['red_days']}+ red days")
        print(f"{'='*60}")

        all_trades = []
        ticker_breakdown = {}

        for ticker in var_cfg['tickers']:
            if ticker not in data:
                print(f"  {ticker}: no data available, skipping")
                ticker_breakdown[ticker] = {'status': 'no_data', 'num_trades': 0}
                continue

            df = data[ticker]
            entries = find_entries(df, var_cfg['dip_pct'], var_cfg['rsi_thresh'], var_cfg['red_days'])
            print(f"  {ticker}: {len(entries)} entry signals found")

            trades = run_backtest_single(
                df, entries, CAPITAL, MAX_PER_TRADE, MAX_CONCURRENT,
                HOLD_DAYS, SLIPPAGE_BPS
            )

            # Tag trades with ticker
            for t in trades:
                t['ticker'] = ticker

            ticker_metrics = compute_metrics(trades, CAPITAL)
            ticker_breakdown[ticker] = ticker_metrics
            all_trades.extend(trades)
            print(f"    -> {ticker_metrics['num_trades']} trades, "
                  f"Sharpe={ticker_metrics['sharpe']}, WR={ticker_metrics['win_rate']}, "
                  f"PF={ticker_metrics['profit_factor']}, DD={ticker_metrics['max_drawdown']}")

        # Sort all trades by entry date for combined portfolio simulation
        all_trades.sort(key=lambda t: t['entry_date'])

        # Combined metrics
        combined_metrics = compute_metrics(all_trades, CAPITAL)

        # Regime analysis
        bull_trades, bear_trades = regime_split(all_trades, spy_df)
        bull_metrics = compute_metrics(bull_trades, CAPITAL)
        bear_metrics = compute_metrics(bear_trades, CAPITAL)

        bull_sharpe = bull_metrics['sharpe']
        bear_sharpe = bear_metrics['sharpe']
        max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
        regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 1e-9 else 0.0

        # Permutation test
        perm_p = permutation_test(all_trades, len(data.get(var_cfg['tickers'][0], pd.DataFrame())), CAPITAL, N_PERMUTATIONS)

        # Five-gate check
        gates = five_gate_check(combined_metrics, regime_gap, perm_p)

        print(f"\n  COMBINED: {combined_metrics['num_trades']} trades")
        print(f"    Sharpe={combined_metrics['sharpe']}, Sortino={combined_metrics['sortino']}")
        print(f"    WR={combined_metrics['win_rate']}, PF={combined_metrics['profit_factor']}")
        print(f"    MaxDD={combined_metrics['max_drawdown']}, TotalReturn={combined_metrics['total_return']}")
        print(f"    Bull Sharpe={bull_sharpe}, Bear Sharpe={bear_sharpe}, Regime Gap={regime_gap:.4f}")
        print(f"    Permutation p={perm_p:.4f}")
        print(f"    5-Gate: {'PASS' if gates['all_pass'] else 'FAIL'} — {gates}")

        results[var_key] = {
            'name': var_cfg['name'],
            'tickers': var_cfg['tickers'],
            'params': {
                'dip_pct': var_cfg['dip_pct'],
                'rsi_thresh': var_cfg['rsi_thresh'],
                'red_days': var_cfg['red_days'],
                'hold_days': HOLD_DAYS,
                'slippage_bps': SLIPPAGE_BPS,
                'max_per_trade': MAX_PER_TRADE,
                'max_concurrent': MAX_CONCURRENT,
            },
            **combined_metrics,
            'bull_sharpe': round(float(bull_sharpe), 4),
            'bear_sharpe': round(float(bear_sharpe), 4),
            'bull_trades': bull_metrics['num_trades'],
            'bear_trades': bear_metrics['num_trades'],
            'regime_gap': round(float(regime_gap), 4),
            'permutation_p': round(float(perm_p), 4),
            'five_gate': gates,
            'ticker_breakdown': ticker_breakdown,
        }

    # Save results
    output_path = Path('/home/jupiter/Lvl3Quant/data/crypto_mr_results.json')
    output = {
        'metadata': {
            'strategy': 'Crypto Mean Reversion — Dual Signal D',
            'period': f'{START} to {END}',
            'capital': CAPITAL,
            'generated': datetime.now().isoformat(),
        },
        'variants': results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Var':<4} {'Name':<35} {'#Tr':>4} {'Sharpe':>7} {'Sort':>7} {'WR':>6} {'PF':>6} {'DD':>7} {'Ret':>7} {'5G':>5}")
    print("-" * 90)
    for k, v in results.items():
        gate_str = "PASS" if v['five_gate']['all_pass'] else "FAIL"
        print(f"{k:<4} {v['name']:<35} {v['num_trades']:>4} {v['sharpe']:>7.3f} {v['sortino']:>7.3f} "
              f"{v['win_rate']:>6.1%} {v['profit_factor']:>6.2f} {v['max_drawdown']:>7.1%} {v['total_return']:>7.1%} {gate_str:>5}")
    print("=" * 90)


if __name__ == '__main__':
    main()
