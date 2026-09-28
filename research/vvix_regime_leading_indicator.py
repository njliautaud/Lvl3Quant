#!/usr/bin/env python3
"""
VVIX as Leading Regime Indicator for Sector Dip-Buying
======================================================

HYPOTHESIS: VVIX (volatility of VIX) moves BEFORE VIX itself during
regime transitions. By monitoring VVIX changes, we can detect regime
shifts 0-2 days earlier and time sector dip entries better.

Deep research (Aug 2026) found:
  - Multi-input ML systems detect regime shifts with 0-2 day lag
    by watching term structure, VVIX, and put/call ratios BEFORE VIX moves
  - Regime-aware ML reduces forecast error by 20% and improves Sharpe by 0.5+
  - VVIX spikes often precede VIX spikes by 1-2 days

WHAT WE TEST:
  A: VVIX spike (>1 std above 20d mean) → regime fear building → wait for dip, then buy
  B: VVIX collapse (drops >1 std in 2 days) → complacency → be cautious
  C: VVIX/VIX ratio divergence → VVIX rising but VIX flat → storm coming
  D: VVIX percentile rank as entry filter (only dip-buy when VVIX elevated)
  E: VVIX mean reversion → buy dips when VVIX is reverting from extreme
  F: Combined: VVIX spike + RSI<35 sector → enhanced dip-buy timing
  G: VVIX acceleration (2d change of VVIX > 10) → rapid fear buildup
  H: VVIX term structure proxy (VVIX vs VIX 5d MA spread)

VALIDATION: Full adversarial battery per HC #428:
  - Sharpe > 0.5, permutation p < 0.05
  - Regime gap < 0.50 (green vs red)
  - Cost robustness to 20bps
  - 4 sub-period consistency

Output: output/growth_research/vvix_regime_leading_v1/
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

OUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/vvix_regime_leading_v1"
os.makedirs(OUT_DIR, exist_ok=True)

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
STARTING_CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0001  # 1 bps
HOLD_DAYS = 5
TP_PCT = 0.03  # 3% take-profit

def download_data():
    """Download VVIX, VIX, SPY, and sector ETF data."""
    print("Downloading data...")

    # Date range: 2018 to present (VVIX has good history from ~2014)
    start = "2018-01-01"
    end = datetime.now().strftime("%Y-%m-%d")

    tickers = ['^VVIX', '^VIX', 'SPY'] + SECTOR_ETFS
    data = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if len(df) > 100:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker.replace('^', '')] = df
                print(f"  {ticker}: {len(df)} days")
            else:
                print(f"  {ticker}: insufficient data ({len(df)} days)")
        except Exception as e:
            print(f"  {ticker}: ERROR {e}")

    return data


def compute_vvix_features(data):
    """Compute VVIX-derived features for regime detection."""
    vvix = data['VVIX']['Close'].copy()
    vix = data['VIX']['Close'].copy()
    spy = data['SPY']['Close'].copy()

    # Align all to common dates
    common = vvix.index.intersection(vix.index).intersection(spy.index)
    vvix = vvix.loc[common]
    vix = vix.loc[common]
    spy = spy.loc[common]

    features = pd.DataFrame(index=common)
    features['vvix'] = vvix
    features['vix'] = vix
    features['spy'] = spy

    # VVIX features
    features['vvix_20d_mean'] = vvix.rolling(20).mean()
    features['vvix_20d_std'] = vvix.rolling(20).std()
    features['vvix_zscore'] = (vvix - features['vvix_20d_mean']) / features['vvix_20d_std']
    features['vvix_2d_change'] = vvix.diff(2)
    features['vvix_5d_change'] = vvix.diff(5)
    features['vvix_pctile'] = vvix.rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

    # VIX features
    features['vix_20d_mean'] = vix.rolling(20).mean()
    features['vix_5d_ma'] = vix.rolling(5).mean()

    # VVIX/VIX ratio
    features['vvix_vix_ratio'] = vvix / vix
    features['vvix_vix_ratio_20d'] = features['vvix_vix_ratio'].rolling(20).mean()
    features['vvix_vix_divergence'] = features['vvix_vix_ratio'] - features['vvix_vix_ratio_20d']

    # VVIX acceleration
    features['vvix_accel'] = features['vvix_2d_change'].diff(2)

    # SPY regime (for validation)
    features['spy_ret_5d'] = spy.pct_change(5)
    features['spy_ret_1d'] = spy.pct_change(1)
    features['spy_green'] = features['spy_ret_1d'] > 0  # For regime stratification

    return features.dropna()


def compute_sector_signals(data, features):
    """Compute RSI and dip signals for each sector ETF."""
    sector_signals = {}

    for etf in SECTOR_ETFS:
        if etf not in data:
            continue

        df = data[etf]['Close'].copy()
        common = df.index.intersection(features.index)
        close = df.loc[common]

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        sector_signals[etf] = {
            'close': close,
            'rsi': rsi,
            'ret_5d': close.pct_change(5),
            'oversold': rsi < 35,
        }

    return sector_signals


def run_variant(variant_name, entry_filter_fn, features, sector_signals, data):
    """Run a single strategy variant with given entry filter."""
    capital = STARTING_CAPITAL
    trades = []
    open_positions = []
    equity_curve = [capital]

    dates = features.index.tolist()

    for i, date in enumerate(dates):
        # Check exits first
        new_open = []
        for pos in open_positions:
            days_held = (date - pos['entry_date']).days
            if pos['etf'] in sector_signals and date in sector_signals[pos['etf']]['close'].index:
                current_price = sector_signals[pos['etf']]['close'].loc[date]
                ret = (current_price / pos['entry_price']) - 1

                # Exit conditions: TP hit or hold period expired
                if ret >= TP_PCT or days_held >= HOLD_DAYS:
                    pnl = pos['shares'] * pos['entry_price'] * ret
                    pnl -= pos['shares'] * pos['entry_price'] * SLIPPAGE_PCT  # exit slippage
                    capital += pos['shares'] * pos['entry_price'] + pnl
                    trades.append({
                        'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                        'exit_date': date.strftime('%Y-%m-%d'),
                        'etf': pos['etf'],
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'ret': ret,
                        'pnl': pnl,
                        'days_held': days_held,
                        'exit_reason': 'tp' if ret >= TP_PCT else 'hold_expired',
                    })
                else:
                    new_open.append(pos)
            else:
                new_open.append(pos)
        open_positions = new_open

        # Check entries
        if len(open_positions) < MAX_CONCURRENT:
            row = features.loc[date]

            # Check if variant's entry filter passes
            if entry_filter_fn(row):
                # Find oversold sectors
                for etf in SECTOR_ETFS:
                    if len(open_positions) >= MAX_CONCURRENT:
                        break
                    if etf not in sector_signals:
                        continue
                    if date not in sector_signals[etf]['rsi'].index:
                        continue

                    rsi_val = sector_signals[etf]['rsi'].loc[date]
                    if rsi_val < 35:  # Oversold
                        # Check not already holding this ETF
                        if any(p['etf'] == etf for p in open_positions):
                            continue

                        price = sector_signals[etf]['close'].loc[date]
                        trade_size = min(MAX_PER_TRADE, capital * 0.33)
                        if trade_size < 10:
                            continue

                        shares = trade_size / price
                        cost = shares * price * (1 + SLIPPAGE_PCT)
                        if cost > capital:
                            continue

                        capital -= cost
                        open_positions.append({
                            'entry_date': date,
                            'etf': etf,
                            'entry_price': price,
                            'shares': shares,
                        })

        # Mark-to-market for equity curve
        mtm = capital
        for pos in open_positions:
            if pos['etf'] in sector_signals and date in sector_signals[pos['etf']]['close'].index:
                current = sector_signals[pos['etf']]['close'].loc[date]
                mtm += pos['shares'] * current
        equity_curve.append(mtm)

    # Close remaining positions at last price
    for pos in open_positions:
        last_date = dates[-1]
        if pos['etf'] in sector_signals and last_date in sector_signals[pos['etf']]['close'].index:
            current = sector_signals[pos['etf']]['close'].loc[last_date]
            ret = (current / pos['entry_price']) - 1
            pnl = pos['shares'] * pos['entry_price'] * ret
            capital += pos['shares'] * pos['entry_price'] + pnl
            trades.append({
                'entry_date': pos['entry_date'].strftime('%Y-%m-%d'),
                'exit_date': last_date.strftime('%Y-%m-%d'),
                'etf': pos['etf'],
                'entry_price': pos['entry_price'],
                'exit_price': current,
                'ret': ret,
                'pnl': pnl,
                'days_held': (last_date - pos['entry_date']).days,
                'exit_reason': 'close_remaining',
            })

    return trades, equity_curve


def compute_metrics(trades, equity_curve, features):
    """Compute Sharpe, Sortino, WR, PF, MaxDD, regime stratification."""
    if not trades:
        return {'sharpe': 0, 'n_trades': 0, 'status': 'no_trades'}

    returns = [t['ret'] for t in trades]
    pnls = [t['pnl'] for t in trades]

    n = len(trades)
    wins = sum(1 for r in returns if r > 0)
    wr = wins / n

    avg_ret = np.mean(returns)
    std_ret = np.std(returns) if np.std(returns) > 0 else 1e-10

    # Annualize (assume avg 5-day hold, ~50 trades/year)
    trades_per_year = 252 / 5  # rough
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year)

    neg_rets = [r for r in returns if r < 0]
    downside_std = np.std(neg_rets) if neg_rets else 1e-10
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    total_pnl = sum(pnls)
    total_ret_pct = (total_pnl / STARTING_CAPITAL) * 100

    # Max drawdown from equity curve
    peak = equity_curve[0]
    max_dd = 0
    for v in equity_curve:
        peak = max(peak, v)
        dd = (v - peak) / peak
        max_dd = min(max_dd, dd)

    # Regime stratification (green vs red SPY days at entry)
    green_rets = []
    red_rets = []
    for t in trades:
        entry = pd.Timestamp(t['entry_date'])
        if entry in features.index:
            if features.loc[entry, 'spy_green']:
                green_rets.append(t['ret'])
            else:
                red_rets.append(t['ret'])

    green_sharpe = 0
    red_sharpe = 0
    if green_rets and np.std(green_rets) > 0:
        green_sharpe = (np.mean(green_rets) / np.std(green_rets)) * np.sqrt(trades_per_year)
    if red_rets and np.std(red_rets) > 0:
        red_sharpe = (np.mean(red_rets) / np.std(red_rets)) * np.sqrt(trades_per_year)

    regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

    # Sub-period analysis (4 equal periods)
    period_sharpes = []
    sorted_trades = sorted(trades, key=lambda t: t['entry_date'])
    chunk = max(1, len(sorted_trades) // 4)
    for j in range(4):
        subset = sorted_trades[j*chunk:(j+1)*chunk]
        if subset:
            sub_rets = [t['ret'] for t in subset]
            sub_std = np.std(sub_rets) if np.std(sub_rets) > 0 else 1e-10
            sub_sharpe = (np.mean(sub_rets) / sub_std) * np.sqrt(trades_per_year)
            period_sharpes.append(sub_sharpe)

    sub_period_all_positive = all(s > 0 for s in period_sharpes) if period_sharpes else False

    # Permutation test (100 shuffles)
    perm_count = 0
    observed_sharpe = sharpe
    for _ in range(1000):
        shuffled = np.random.permutation(returns)
        s_std = np.std(shuffled) if np.std(shuffled) > 0 else 1e-10
        s_sharpe = (np.mean(shuffled) / s_std) * np.sqrt(trades_per_year)
        if s_sharpe >= observed_sharpe:
            perm_count += 1
    perm_p = perm_count / 1000

    # Cost robustness (test at 20bps)
    cost_rets = [r - 0.002 for r in returns]  # 20bps cost
    cost_std = np.std(cost_rets) if np.std(cost_rets) > 0 else 1e-10
    cost_sharpe = (np.mean(cost_rets) / cost_std) * np.sqrt(trades_per_year)

    # Gates
    gates = {
        'sharpe_gt_0.5': sharpe > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.50': regime_gap < 0.50,
        'cost_robust_20bps': cost_sharpe > 0,
        'sub_periods_positive': sub_period_all_positive,
    }

    return {
        'n_trades': n,
        'win_rate': round(wr, 4),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 2),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'total_return_pct': round(total_ret_pct, 2),
        'total_pnl': round(total_pnl, 2),
        'avg_return': round(avg_ret * 100, 4),
        'green_sharpe': round(green_sharpe, 3),
        'red_sharpe': round(red_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'perm_p': round(perm_p, 4),
        'cost_sharpe_20bps': round(cost_sharpe, 3),
        'sub_period_sharpes': [round(s, 2) for s in period_sharpes],
        'green_trades': len(green_rets),
        'red_trades': len(red_rets),
        'gates': gates,
        'gates_passed': sum(v for v in gates.values()),
    }


def main():
    print("=" * 60)
    print("VVIX Leading Regime Indicator — Sector Dip-Buy Enhancement")
    print("=" * 60)

    data = download_data()

    if 'VVIX' not in data or 'VIX' not in data:
        print("ERROR: Missing VVIX or VIX data")
        return

    features = compute_vvix_features(data)
    sector_signals = compute_sector_signals(data, features)

    print(f"\nFeature period: {features.index[0].strftime('%Y-%m-%d')} to {features.index[-1].strftime('%Y-%m-%d')}")
    print(f"VVIX range: {features['vvix'].min():.1f} - {features['vvix'].max():.1f}, mean: {features['vvix'].mean():.1f}")
    print(f"VIX range: {features['vix'].min():.1f} - {features['vix'].max():.1f}")

    # Define variants
    variants = {
        'A_VVIX_Spike': lambda row: row['vvix_zscore'] > 1.0,
        'B_VVIX_Collapse': lambda row: row['vvix_2d_change'] < -5,
        'C_VVIX_VIX_Divergence': lambda row: row['vvix_vix_divergence'] > 0.3,
        'D_VVIX_High_Pctile': lambda row: row['vvix_pctile'] > 0.75,
        'E_VVIX_Mean_Revert': lambda row: row['vvix_zscore'] > 1.5 and row['vvix_2d_change'] < 0,
        'F_VVIX_Spike_RSI': lambda row: row['vvix_zscore'] > 0.5,  # RSI<35 already checked in trade logic
        'G_VVIX_Acceleration': lambda row: row['vvix_accel'] > 5,
        'H_VVIX_Spread': lambda row: (row['vvix'] - row['vix_5d_ma'] * 5.5) > 5,
        'BASELINE_Always': lambda row: True,  # Baseline: dip-buy whenever RSI<35 (no VVIX filter)
    }

    results = {}

    for name, filter_fn in variants.items():
        print(f"\nRunning {name}...")
        trades, equity = run_variant(name, filter_fn, features, sector_signals, data)
        metrics = compute_metrics(trades, equity, features)
        results[name] = metrics

        passed = metrics.get('gates_passed', 0)
        total = len(metrics.get('gates', {}))
        print(f"  Trades: {metrics['n_trades']}, WR: {metrics.get('win_rate',0)*100:.1f}%, "
              f"Sharpe: {metrics.get('sharpe',0):.3f}, PF: {metrics.get('profit_factor',0):.2f}, "
              f"MaxDD: {metrics.get('max_drawdown_pct',0):.1f}%, "
              f"Regime gap: {metrics.get('regime_gap',0):.3f}, "
              f"Gates: {passed}/{total}")

    # Save results
    output = {
        'metadata': {
            'backtest': 'VVIX Leading Regime Indicator for Sector Dip-Buying',
            'period': f"{features.index[0].strftime('%Y-%m-%d')} to {features.index[-1].strftime('%Y-%m-%d')}",
            'starting_capital': STARTING_CAPITAL,
            'hold_days': HOLD_DAYS,
            'tp_pct': TP_PCT,
            'slippage_bps': SLIPPAGE_PCT * 10000,
            'run_date': datetime.now().isoformat(),
        },
        'results': results,
    }

    with open(os.path.join(OUT_DIR, 'results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — Sorted by Sharpe")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)
    for name, m in sorted_results:
        passed = m.get('gates_passed', 0)
        total = len(m.get('gates', {}))
        status = "✅" if passed >= 4 else "🟡" if passed >= 3 else "❌"
        print(f"{status} {name}: Sharpe {m.get('sharpe',0):.2f} | WR {m.get('win_rate',0)*100:.0f}% | "
              f"PF {m.get('profit_factor',0):.1f} | DD {m.get('max_drawdown_pct',0):.1f}% | "
              f"Trades {m['n_trades']} | Regime {m.get('regime_gap',0):.2f} | "
              f"Gates {passed}/{total}")

    print(f"\nResults saved to {OUT_DIR}/results.json")


if __name__ == '__main__':
    main()
