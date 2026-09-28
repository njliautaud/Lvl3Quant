#!/usr/bin/env python3
"""
International Quality Mean Reversion Backtest
==============================================
Tests whether the validated QMR dual-signal logic works on international quality
stocks accessible via US-listed ADRs and country ETFs.

Dual Signal D logic:
  Signal 1: 5%+ dip from 20d high AND RSI(14) < 35
  Signal 2: First green day after 3+ consecutive red days (post 5% drawdown)
  BOTH must fire on same date+ticker to trigger entry. Hold 10 days.

6 Variants:
  A: ADRs only (15 stocks)
  B: Country ETFs only (5 ETFs)
  C: Combined ADRs + country ETFs
  D: ADRs only, 7% dip threshold (ADRs more volatile)
  E: ADRs only, with VIX < 25 filter
  F: Top 10 ADRs by market cap only

OOT: Jan 2022 - Jul 2026 | Capital: $645 | Max $200/trade | Max 3 concurrent
Slippage: 0.03% each way (higher for ADRs - wider spreads)
5-Gate Validation + Permutation Test (1000 iterations)
"""

import json
import warnings
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────────
ADRS = [
    'TSM', 'ASML', 'NVO', 'SAP', 'TM', 'SNY', 'AZN', 'SHOP',
    'NVS', 'DEO', 'UL', 'SONY', 'MUFG', 'BHP', 'RIO',
]
COUNTRY_ETFS = ['EWJ', 'EWG', 'EWU', 'EWC', 'EWA']
TOP10_ADRS = ['TSM', 'ASML', 'NVO', 'SAP', 'AZN', 'NVS', 'SHOP', 'TM', 'SONY', 'BHP']

SPY = 'SPY'
VIX_TICKER = '^VIX'
START_DATE = '2021-01-01'  # extra lookback for indicators
OOT_START = '2022-01-01'
OOT_END = '2026-07-31'
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0003  # 0.03% each way (ADRs have wider spreads)
MAX_POSITIONS = 3
MAX_PER_TRADE = 200.0
HOLD_DAYS = 10
N_PERMUTATIONS = 1000
VIX_THRESHOLD = 25

RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/international_quality_mr_results.json'

# Variant definitions
VARIANT_CONFIGS = {
    'A': {
        'desc': 'ADRs only (15 stocks), dual signal D',
        'universe': ADRS,
        'dip_threshold': 0.05,
        'vix_filter': False,
    },
    'B': {
        'desc': 'Country ETFs only (5 ETFs), dual signal D',
        'universe': COUNTRY_ETFS,
        'dip_threshold': 0.05,
        'vix_filter': False,
    },
    'C': {
        'desc': 'Combined ADRs + country ETFs, dual signal D',
        'universe': ADRS + COUNTRY_ETFS,
        'dip_threshold': 0.05,
        'vix_filter': False,
    },
    'D': {
        'desc': 'ADRs only, 7% dip threshold (more volatile)',
        'universe': ADRS,
        'dip_threshold': 0.07,
        'vix_filter': False,
    },
    'E': {
        'desc': 'ADRs only, VIX < 25 filter',
        'universe': ADRS,
        'dip_threshold': 0.05,
        'vix_filter': True,
    },
    'F': {
        'desc': 'Top 10 ADRs by market cap, dual signal D',
        'universe': TOP10_ADRS,
        'dip_threshold': 0.05,
        'vix_filter': False,
    },
}


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download all price data."""
    all_tickers = sorted(set(ADRS + COUNTRY_ETFS + [SPY]))
    print(f"Downloading price data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END,
                       auto_adjust=True, progress=False)

    close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
    close = close.ffill().dropna(how='all')

    # Download VIX separately (special ticker)
    print("Downloading VIX data...")
    vix_data = yf.download(VIX_TICKER, start=START_DATE, end=OOT_END,
                           auto_adjust=True, progress=False)
    vix_close = vix_data['Close'].squeeze() if len(vix_data) > 0 else pd.Series(dtype=float)

    # Check what we got
    available = [t for t in all_tickers if t in close.columns]
    missing = [t for t in all_tickers if t not in close.columns]
    print(f"  Available: {len(available)}/{len(all_tickers)} tickers")
    if missing:
        print(f"  Missing: {missing}")
    print(f"  Date range: {close.index[0].date()} to {close.index[-1].date()}")
    print(f"  VIX rows: {len(vix_close)}")

    return close, vix_close


# ── Indicator Helpers ──────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Compute RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def consecutive_red_days(series):
    """Count consecutive red (close < prior close) days ending at each date."""
    is_red = (series.diff() < 0).astype(int)
    result = pd.Series(0, index=series.index, dtype=int)
    count = 0
    for i in range(len(is_red)):
        if is_red.iloc[i] == 1:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result


def compute_indicators(close, vix_close):
    """Pre-compute all indicators."""
    indicators = {}
    all_universe = sorted(set(ADRS + COUNTRY_ETFS))

    for t in all_universe:
        if t not in close.columns:
            continue
        s = close[t].dropna()
        if len(s) < 50:
            continue
        indicators[t] = {
            'close': s,
            'rsi14': compute_rsi(s, 14),
            'high20': s.rolling(20).max(),
            'consec_red': consecutive_red_days(s),
            'green': s.diff() > 0,
        }

    # SPY for regime classification
    if SPY in close.columns:
        indicators['SPY_close'] = close[SPY]
        indicators['SPY_200SMA'] = close[SPY].rolling(200).mean()

    indicators['VIX'] = vix_close

    print(f"  Indicators computed for {sum(1 for t in all_universe if t in indicators)} tickers")
    return indicators


def get_regime(date, indicators):
    """Bull if SPY > 200-SMA, else Bear."""
    spy_close = indicators.get('SPY_close')
    spy_sma = indicators.get('SPY_200SMA')
    if spy_close is None or spy_sma is None:
        return 'unknown'
    try:
        sc = spy_close.asof(date)
        ss = spy_sma.asof(date)
        if pd.notna(sc) and pd.notna(ss):
            return 'bull' if sc > ss else 'bear'
    except Exception:
        pass
    return 'unknown'


def get_vix(date, indicators):
    """Get VIX value for a date."""
    vix = indicators.get('VIX')
    if vix is None:
        return 20.0
    try:
        v = vix.asof(date)
        return float(v) if pd.notna(v) else 20.0
    except Exception:
        return 20.0


# ── Signal Generation (Dual Signal D) ─────────────────────────────────────
def generate_dual_signals(date, indicators, universe, dip_threshold=0.05):
    """Generate Dual Signal D: BOTH conditions must fire on same date+ticker.

    Signal 1 (QMR-A): drop > dip_threshold from 20d high AND RSI(14) < 35
    Signal 2 (Recovery): first green day after 3+ consecutive red days, post drawdown

    Returns list of (ticker, price) tuples where BOTH signals agree.
    """
    signals = []
    for t in universe:
        if t not in indicators:
            continue
        ind = indicators[t]
        c = ind['close']
        if date not in c.index:
            continue

        try:
            price = float(c.loc[date])
            high20 = float(ind['high20'].loc[date])
            rsi = float(ind['rsi14'].loc[date])
        except (KeyError, TypeError):
            continue

        if pd.isna(price) or pd.isna(high20) or pd.isna(rsi) or high20 == 0:
            continue

        drawdown = (price - high20) / high20

        # Signal 1: QMR-A dip buy
        signal1 = drawdown < -dip_threshold and rsi < 35

        # Signal 2: First green after 3+ red days
        signal2 = False
        loc = c.index.get_loc(date)
        if loc >= 1:
            try:
                is_green = bool(ind['green'].iloc[loc])
                prev_consec = int(ind['consec_red'].iloc[loc - 1])
                # Also require drawdown for signal 2
                if is_green and prev_consec >= 3 and drawdown < -dip_threshold:
                    signal2 = True
            except (IndexError, ValueError):
                pass

        # Dual Signal D: BOTH must fire
        if signal1 and signal2:
            signals.append((t, price))

    return signals


# ── Trade Simulator ────────────────────────────────────────────────────────
def run_backtest(close, indicators, variant_cfg):
    """Run backtest for a single variant."""
    universe = variant_cfg['universe']
    dip_threshold = variant_cfg['dip_threshold']
    vix_filter = variant_cfg['vix_filter']

    oot_dates = close.loc[OOT_START:OOT_END].index
    if len(oot_dates) == 0:
        return [], pd.Series(dtype=float)

    capital = STARTING_CAPITAL
    positions = []  # {ticker, entry_price, entry_date, shares, exit_target, regime}
    trades = []
    equity = {}

    for date in oot_dates:
        # Exit positions that hit hold period
        still_open = []
        for pos in positions:
            if date >= pos['exit_target']:
                try:
                    exit_price_raw = close.loc[date, pos['ticker']]
                except (KeyError, TypeError):
                    exit_price_raw = pos['entry_price']
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
                })
            else:
                still_open.append(pos)
        positions = still_open

        # VIX filter check
        if vix_filter and get_vix(date, indicators) >= VIX_THRESHOLD:
            # Still track equity but don't enter new trades
            mtm = capital
            for pos in positions:
                try:
                    curr = float(close.loc[date, pos['ticker']])
                except (KeyError, TypeError):
                    curr = pos['entry_price']
                if pd.isna(curr):
                    curr = pos['entry_price']
                mtm += curr * pos['shares']
            equity[date] = mtm
            continue

        # Generate signals and enter new positions
        if len(positions) < MAX_POSITIONS:
            signals = generate_dual_signals(date, indicators, universe, dip_threshold)
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
                idx_loc = oot_dates.get_loc(date)
                exit_target = oot_dates[min(idx_loc + HOLD_DAYS, len(oot_dates) - 1)]
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
            try:
                curr = float(close.loc[date, pos['ticker']])
            except (KeyError, TypeError):
                curr = pos['entry_price']
            if pd.isna(curr):
                curr = pos['entry_price']
            mtm += curr * pos['shares']
        equity[date] = mtm

    # Close remaining positions at last date
    last_date = oot_dates[-1]
    for pos in positions:
        try:
            exit_price_raw = float(close.loc[last_date, pos['ticker']])
        except (KeyError, TypeError):
            exit_price_raw = pos['entry_price']
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


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(trades, equity_series):
    """Compute performance metrics."""
    if len(trades) == 0:
        return {
            'n_trades': 0, 'total_pnl': 0, 'total_return_pct': 0,
            'sharpe': 0, 'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
            'max_drawdown_pct': 0, 'avg_trade_return_pct': 0,
        }

    total_pnl = sum(t['pnl'] for t in trades)
    total_return = (equity_series.iloc[-1] / STARTING_CAPITAL - 1) if len(equity_series) > 0 else 0

    # Daily returns from equity curve for Sharpe/Sortino
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
    winners = sum(1 for t in trades if t['pnl'] > 0)
    win_rate = winners / len(trades)

    # Max drawdown
    if len(equity_series) > 0:
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
    else:
        max_dd = 0

    avg_trade_ret = np.mean([t['return'] for t in trades]) * 100

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
    """Compute Sharpe for bull vs bear regimes."""
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
    """Sign-shuffle permutation test. Returns p-value."""
    if len(trades) < 5 or len(equity_series) < 10:
        return 1.0

    # Observed Sharpe from equity curve
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
    """5-gate validation."""
    gates = {
        'G1_sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'G2_perm_p_lt_0.05': perm_p < 0.05,
        'G3_regime_gap_lt_0.5': regime_metrics['regime_gap'] < 0.5,
        'G4_maxdd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'G5_min_20_trades': metrics['n_trades'] >= 20,
    }
    gates['all_pass'] = all(v for k, v in gates.items())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    close, vix_close = download_data()
    indicators = compute_indicators(close, vix_close)

    results = {}
    summary_rows = []

    for variant_name in ['A', 'B', 'C', 'D', 'E', 'F']:
        cfg = VARIANT_CONFIGS[variant_name]
        print(f"\n{'='*60}")
        print(f"Running Variant {variant_name}: {cfg['desc']}")
        print(f"  Universe: {cfg['universe']}")
        print(f"  Dip threshold: {cfg['dip_threshold']*100:.0f}%, VIX filter: {cfg['vix_filter']}")
        print(f"{'='*60}")

        trades, equity = run_backtest(close, indicators, cfg)
        metrics = compute_metrics(trades, equity)
        regime = compute_regime_sharpe(trades)

        print(f"  Trades: {metrics['n_trades']}, PnL: ${metrics['total_pnl']:.2f}, "
              f"Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}, "
              f"PF: {metrics['profit_factor']:.2f}, WR: {metrics['win_rate']:.1%}")
        print(f"  MaxDD: {metrics['max_drawdown_pct']:.1f}%, AvgTrade: {metrics['avg_trade_return_pct']:.2f}%")
        print(f"  Regime: Bull={regime['sharpe_bull']:.3f} ({regime['n_bull']}), "
              f"Bear={regime['sharpe_bear']:.3f} ({regime['n_bear']}), Gap={regime['regime_gap']:.3f}")

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
        perm_p = permutation_test(trades, equity)
        print(f"  Perm p-value: {perm_p:.4f}")

        gates = five_gate_validation(metrics, regime, perm_p)
        passed = sum(1 for k, v in gates.items() if k != 'all_pass' and v)
        status = 'PASS' if gates['all_pass'] else 'FAIL'
        print(f"  Gates: {passed}/5 {status}")
        for gname, gval in gates.items():
            if gname != 'all_pass':
                print(f"    {gname}: {'PASS' if gval else 'FAIL'}")

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
            'universe': cfg['universe'],
            'dip_threshold': cfg['dip_threshold'],
            'vix_filter': cfg['vix_filter'],
            'hold_days': HOLD_DAYS,
            'metrics': metrics,
            'regime': regime,
            'permutation_p_value': perm_p,
            'gates': {k: bool(v) for k, v in gates.items()},
            'ticker_breakdown': {t: {'pnl': round(p, 2), 'trades': ticker_count[t]} for t, p in top_tickers},
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
            '5Gate': status,
        })

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n{'='*110}")
    print("INTERNATIONAL QUALITY MEAN REVERSION - SUMMARY TABLE")
    print(f"Strategy: Dual Signal D (5%+ dip from 20d high + RSI<35 AND first green after 3+ red)")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL} | "
          f"Slippage: {SLIPPAGE_PCT*100:.2f}%/side | Max {MAX_POSITIONS} concurrent | Hold {HOLD_DAYS}d")
    print(f"{'='*110}")

    df = pd.DataFrame(summary_rows)
    print(df[['Variant', 'Desc', 'Trades', 'PnL', 'Return', 'Sharpe', 'Sortino',
              'PF', 'WR', 'MaxDD', 'RegGap', 'PermP', '5Gate']].to_string(index=False))

    passed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'PASS']
    failed_variants = [r['Variant'] for r in summary_rows if r['5Gate'] == 'FAIL']
    print(f"\nPASSED 5-GATE: {passed_variants if passed_variants else 'None'}")
    print(f"FAILED 5-GATE: {failed_variants if failed_variants else 'None'}")

    # ── Key Findings ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("KEY FINDINGS")
    print(f"{'='*70}")

    # Best variant
    best = max(summary_rows, key=lambda x: x['Sharpe'])
    print(f"  Best Sharpe: Variant {best['Variant']} ({best['Desc']}) = {best['Sharpe']:.3f}")

    # ADR vs ETF comparison
    a_sharpe = results['A']['metrics']['sharpe'] if 'A' in results else 0
    b_sharpe = results['B']['metrics']['sharpe'] if 'B' in results else 0
    print(f"  ADRs (A) Sharpe: {a_sharpe:.3f} vs ETFs (B) Sharpe: {b_sharpe:.3f}")
    if a_sharpe > b_sharpe:
        print(f"  -> Individual ADRs show stronger mean reversion than country ETFs")
    else:
        print(f"  -> Country ETFs show stronger mean reversion than individual ADRs")

    # Volatility adjustment impact
    d_sharpe = results['D']['metrics']['sharpe'] if 'D' in results else 0
    print(f"  7% threshold (D) Sharpe: {d_sharpe:.3f} vs 5% (A) Sharpe: {a_sharpe:.3f}")

    # VIX filter impact
    e_sharpe = results['E']['metrics']['sharpe'] if 'E' in results else 0
    print(f"  VIX<25 filter (E) Sharpe: {e_sharpe:.3f} vs unfiltered (A): {a_sharpe:.3f}")

    # Trade counts
    total_signals = sum(r['Trades'] for r in summary_rows)
    print(f"\n  Total signals across all variants: {total_signals}")
    if total_signals == 0:
        print("  NOTE: Dual Signal D (intersection of both signals) is very selective.")
        print("  The requirement for BOTH signals to fire on the same day may be too strict")
        print("  for international names with different volatility patterns.")

    # ── Save Results ───────────────────────────────────────────────────────
    results['_meta'] = {
        'strategy': 'International Quality Mean Reversion (Dual Signal D)',
        'signal_logic': 'BOTH: (1) 5%+ dip from 20d high + RSI<35, AND (2) first green after 3+ red days',
        'adrs': ADRS,
        'country_etfs': COUNTRY_ETFS,
        'top10_adrs': TOP10_ADRS,
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'starting_capital': STARTING_CAPITAL,
        'slippage_pct_each_way': SLIPPAGE_PCT,
        'max_positions': MAX_POSITIONS,
        'max_per_trade': MAX_PER_TRADE,
        'hold_days': HOLD_DAYS,
        'n_permutations': N_PERMUTATIONS,
        'vix_threshold': VIX_THRESHOLD,
        'run_timestamp': datetime.now().isoformat(),
        'passed_variants': passed_variants,
        'failed_variants': failed_variants,
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")
    print("DONE.")


if __name__ == '__main__':
    main()
