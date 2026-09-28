#!/usr/bin/env python3
"""
TRUE Out-of-Sample Lockbox Validation: ETF Flow Reversal Strategy
=================================================================
Tests the AVO-evolved strategy (v25, score 4.36) on 2026 data NEVER seen
during evolution (evolved on 2022-2025 walk-forward folds).

OOS window: 2026-01-02 to 2026-08-22
Warmup: 2025-07-01 to 2025-12-31 (for fit() + indicator warmup)
Tests: MAX_CONCURRENT=2 (strategy default) AND MAX_CONCURRENT=1
"""

import sys
import os
import json
import importlib.util
import traceback
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── Load strategy ──────────────────────────────────────────────────────────
STRATEGY_PATH = '/home/jupiter/teleclaude-main/runs/flow_reversal-20260824-011411/work/strategy.py'
spec = importlib.util.spec_from_file_location("strategy", STRATEGY_PATH)
strategy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strategy)

# ─── Constants ──────────────────────────────────────────────────────────────
FEATURE_STORE = '/home/jupiter/Lvl3Quant/data/feature_store/sector_etf_flows/daily.parquet'
CAPITAL = 10000.0
OOS_START = '2026-01-02'
OOS_END = '2026-08-22'
WARMUP_START = '2025-01-01'  # enough for fit() + rolling indicators
TRAIN_START = '2025-07-01'   # 6-month train window for fit()
TRAIN_END = '2025-12-31'

ETFS = ['ARKK', 'IGV', 'ITB', 'KRE', 'KWEB', 'OIH', 'SMH', 'SPY',
        'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE',
        'XLU', 'XLV', 'XLY']

VIX_TICKER = '^VIX'
BENCHMARK = 'SPY'

# ─── Build feature data ────────────────────────────────────────────────────
def build_feature_data():
    """
    Load feature store + extend with yfinance through Aug 2026.
    Recompute derived columns for any new data.
    """
    print("Loading feature store...")
    fs = pd.read_parquet(FEATURE_STORE)
    fs['date'] = pd.to_datetime(fs['date'])
    fs_max_date = fs['date'].max()
    print(f"  Feature store covers through {fs_max_date.date()}")

    # Download fresh price data to extend beyond feature store
    import yfinance as yf
    print("Downloading price data (2025-01 through 2026-08-23)...")
    tickers = ETFS + [VIX_TICKER]
    raw = yf.download(tickers, start='2025-01-01', end='2026-08-23',
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
        volume = raw['Volume']
    else:
        close = raw[['Close']].copy()
        volume = raw[['Volume']].copy()

    # Flatten MultiIndex if needed
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = [c[0] if isinstance(c, tuple) else c for c in close.columns]
    if isinstance(volume.columns, pd.MultiIndex):
        volume.columns = [c[0] if isinstance(c, tuple) else c for c in volume.columns]

    close = close.ffill().dropna(how='all')
    volume = volume.ffill().fillna(0)

    # Build extended feature data for dates beyond feature store
    spy_close = close[BENCHMARK] if BENCHMARK in close.columns else None

    new_rows = []
    for etf in ETFS:
        if etf not in close.columns or etf not in volume.columns:
            print(f"  WARNING: {etf} missing from yfinance data")
            continue

        etf_close = close[etf].dropna()
        etf_volume = volume[etf].dropna()

        # Only add dates after feature store ends
        new_dates = etf_close.index[etf_close.index > fs_max_date]
        if len(new_dates) == 0:
            continue

        # Need full history for return calculations
        full_close = etf_close.copy()

        for dt in new_dates:
            c = full_close.loc[dt]
            v = etf_volume.loc[dt] if dt in etf_volume.index else 0

            # ret_1d
            prev_dates = full_close.index[full_close.index < dt]
            ret_1d = np.nan
            if len(prev_dates) > 0:
                prev_c = full_close.loc[prev_dates[-1]]
                ret_1d = (c - prev_c) / prev_c if prev_c > 0 else 0.0

            # ret_20d
            ret_20d = np.nan
            lookback_20 = full_close.index[full_close.index <= dt]
            if len(lookback_20) > 20:
                p20 = full_close.loc[lookback_20[-21]]
                ret_20d = (c - p20) / p20 if p20 > 0 else 0.0

            # ret_60d
            ret_60d = np.nan
            if len(lookback_20) > 60:
                p60 = full_close.loc[lookback_20[-61]]
                ret_60d = (c - p60) / p60 if p60 > 0 else 0.0

            # rel_strength_spy
            rel_strength = np.nan
            if spy_close is not None and dt in spy_close.index:
                spy_lookback = spy_close.index[spy_close.index <= dt]
                if len(spy_lookback) > 20 and len(lookback_20) > 20:
                    spy_ret20 = (spy_close.loc[dt] - spy_close.loc[spy_lookback[-21]]) / spy_close.loc[spy_lookback[-21]]
                    rel_strength = (ret_20d - spy_ret20) if not np.isnan(ret_20d) else np.nan

            new_rows.append({
                'etf': etf,
                'date': dt,
                'close': c,
                'volume': v,
                'dollar_volume': c * v,
                'shares_out': 0,
                'aum_proxy': 0,
                'ret_1d': ret_1d,
                'ret_20d': ret_20d,
                'ret_60d': ret_60d,
                'rel_strength_spy': rel_strength,
            })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        new_df['date'] = pd.to_datetime(new_df['date'])
        combined = pd.concat([fs, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['etf', 'date'], keep='last')
        combined = combined.sort_values(['etf', 'date']).reset_index(drop=True)
        print(f"  Extended to {combined['date'].max().date()} (+{len(new_rows)} rows)")
    else:
        combined = fs
        print("  No extension needed")

    # Also return prices for trade simulation
    return combined, close, volume


def run_backtest(feature_data, prices, max_concurrent_override=None):
    """
    Run full backtest on 2026 OOS data.

    1. fit() on TRAIN_START..TRAIN_END
    2. generate_signals() on full OOS window
    3. Simulate trades with next-day entry
    """
    max_concurrent = max_concurrent_override or strategy.MAX_CONCURRENT
    max_per_trade = strategy.MAX_PER_TRADE
    slippage_pct = strategy.SLIPPAGE_PCT

    # ─── Train phase ────────────────────────────────────────────────────────
    train_mask = (feature_data['date'] >= TRAIN_START) & (feature_data['date'] <= TRAIN_END)
    train_data = feature_data[train_mask].copy()
    print(f"  Training on {len(train_data)} rows ({TRAIN_START} to {TRAIN_END})")

    params = strategy.fit(train_data)
    tradeable = params.get('tradeable_etfs', [])
    print(f"  Tradeable ETFs from fit(): {len(tradeable)} — {tradeable[:10]}...")

    # ─── Signal generation on OOS window ────────────────────────────────────
    # Include warmup data so rolling indicators are computed correctly
    signal_mask = (feature_data['date'] >= WARMUP_START) & (feature_data['date'] <= OOS_END)
    signal_data = feature_data[signal_mask].copy()
    print(f"  Generating signals on {len(signal_data)} rows (with warmup)")

    signals = strategy.generate_signals(signal_data, params)
    if signals is None or len(signals) == 0:
        print("  WARNING: No signals generated!")
        return None, [], None

    # Filter signals to OOS window only
    signals['date'] = pd.to_datetime(signals['date'])
    oos_signals = signals[signals['date'] >= OOS_START].copy()
    print(f"  Signals in OOS window: {len(oos_signals)}")

    # Build signal lookup: date -> [(etf, score)]
    signal_lookup = {}
    for _, row in oos_signals.iterrows():
        d = pd.Timestamp(row['date'])
        if d not in signal_lookup:
            signal_lookup[d] = []
        signal_lookup[d].append((row['etf'], row.get('score', 1.0)))

    # ─── Trade simulation ───────────────────────────────────────────────────
    oos_dates_mask = (prices.index >= OOS_START) & (prices.index <= OOS_END)
    dates = prices.index[oos_dates_mask]
    if len(dates) == 0:
        print("  ERROR: No trading dates in OOS window!")
        return None, [], None

    capital = CAPITAL
    trades = []
    open_positions = []
    daily_pnl = pd.Series(0.0, index=dates)
    equity_curve = pd.Series(CAPITAL, index=dates)
    peak_equity = CAPITAL
    pending_entries = []

    for i, date in enumerate(dates):
        curr_equity = equity_curve.iloc[max(0, i-1)] if i > 0 else CAPITAL
        portfolio_dd = (curr_equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

        # ─── Process exits ──────────────────────────────────────────────────
        new_open = []
        for pos in open_positions:
            ticker = pos['ticker']
            if ticker not in prices.columns:
                new_open.append(pos)
                continue

            curr_price = prices.loc[date, ticker]
            if pd.isna(curr_price):
                new_open.append(pos)
                continue

            # Update high water mark
            if curr_price > pos.get('high_water_mark', pos['entry_price_adj']):
                pos['high_water_mark'] = curr_price

            do_exit = strategy.should_exit(pos, curr_price, date, portfolio_dd)

            if do_exit:
                exit_price = curr_price * (1.0 - slippage_pct)
                pnl = (exit_price - pos['entry_price_adj']) * pos['shares']
                capital += pos['size_dollars'] + pnl
                daily_pnl.iloc[i] += pnl

                entry_p = pos['entry_price_adj']
                pnl_pct = (curr_price - entry_p) / entry_p
                days_held = int(np.busday_count(
                    np.datetime64(pos['entry_date'], 'D'),
                    np.datetime64(date, 'D')))

                # Classify exit reason
                if pnl_pct >= strategy.TAKE_PROFIT_PCT:
                    exit_reason = 'take_profit'
                elif days_held >= strategy.MAX_HOLD_DAYS:
                    exit_reason = 'max_hold'
                elif pnl_pct < strategy.UNDERWATER_TOLERANCE and days_held >= strategy.UNDERWATER_CUT_DAYS:
                    exit_reason = 'underwater_cut'
                elif portfolio_dd < -0.01 and pnl_pct < 0.003:
                    exit_reason = 'portfolio_dd'
                else:
                    exit_reason = 'trailing_stop'

                trades.append({
                    'ticker': ticker,
                    'entry_date': str(pos['entry_date'])[:10],
                    'exit_date': str(date)[:10],
                    'entry_price': round(float(pos['entry_price_adj']), 4),
                    'exit_price': round(float(exit_price), 4),
                    'shares': pos['shares'],
                    'pnl': round(float(pnl), 2),
                    'return_pct': round(float(pnl / pos['size_dollars'] * 100), 4),
                    'days_held': days_held,
                    'exit_reason': exit_reason,
                })
            else:
                # Mark-to-market for equity curve
                if i > 0:
                    prev_price = prices.loc[dates[i-1], ticker]
                    if not pd.isna(prev_price):
                        daily_pnl.iloc[i] += (curr_price - prev_price) * pos['shares']
                new_open.append(pos)

        open_positions = new_open

        # ─── Process pending entries (next-day execution) ───────────────────
        for ticker, score in pending_entries:
            if len(open_positions) >= max_concurrent:
                break
            if any(p['ticker'] == ticker for p in open_positions):
                continue
            if ticker not in prices.columns:
                continue

            price = prices.loc[date, ticker]
            if pd.isna(price) or price <= 0:
                continue

            size = min(max_per_trade, capital * 0.95)
            if size < 10:
                continue

            entry_price = price * (1.0 + slippage_pct)
            shares = int(size / entry_price)
            if shares < 1:
                continue

            actual_cost = shares * entry_price
            capital -= actual_cost
            open_positions.append({
                'ticker': ticker,
                'entry_date': date,
                'entry_price_adj': entry_price,
                'high_water_mark': entry_price,
                'shares': shares,
                'size_dollars': actual_cost,
            })

        pending_entries = []

        # ─── Check for new signals (execute NEXT day) ───────────────────────
        if date in signal_lookup and len(open_positions) < max_concurrent:
            candidates = sorted(signal_lookup[date], key=lambda x: x[1], reverse=True)
            for ticker, score in candidates:
                if not any(p['ticker'] == ticker for p in open_positions):
                    pending_entries.append((ticker, score))

        # ─── Update equity curve ────────────────────────────────────────────
        if i > 0:
            equity_curve.iloc[i] = equity_curve.iloc[i-1] + daily_pnl.iloc[i]
        else:
            equity_curve.iloc[i] = CAPITAL + daily_pnl.iloc[i]

        peak_equity = max(peak_equity, equity_curve.iloc[i])

    # ─── Force close remaining ──────────────────────────────────────────────
    if open_positions and len(dates) > 0:
        last_date = dates[-1]
        for pos in open_positions:
            ticker = pos['ticker']
            if ticker not in prices.columns:
                continue
            price = prices.loc[last_date, ticker]
            if pd.isna(price):
                continue
            exit_price = price * (1.0 - slippage_pct)
            pnl = (exit_price - pos['entry_price_adj']) * pos['shares']
            days_held = int(np.busday_count(
                np.datetime64(pos['entry_date'], 'D'),
                np.datetime64(last_date, 'D')))
            trades.append({
                'ticker': ticker,
                'entry_date': str(pos['entry_date'])[:10],
                'exit_date': str(last_date)[:10],
                'entry_price': round(float(pos['entry_price_adj']), 4),
                'exit_price': round(float(exit_price), 4),
                'shares': pos['shares'],
                'pnl': round(float(pnl), 2),
                'return_pct': round(float(pnl / pos['size_dollars'] * 100), 4),
                'days_held': days_held,
                'exit_reason': 'force_close',
            })

    return equity_curve, trades, daily_pnl


def compute_metrics(trades, daily_pnl, equity_curve):
    """Compute risk-adjusted performance metrics."""
    n = len(trades)
    if n == 0:
        return {
            'total_trades': 0, 'win_rate': 0, 'profit_factor': 0,
            'sharpe': 0, 'sortino': 0, 'total_return_pct': 0,
            'total_pnl': 0, 'max_drawdown_pct': 0, 'avg_days_held': 0,
        }

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / n

    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    daily_ret = daily_pnl / CAPITAL
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)

    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std()
    sharpe = float(mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    downside = daily_ret[daily_ret < 0]
    ds_std = downside.std() if len(downside) > 5 else std_ret
    sortino = float(mean_ret / ds_std * np.sqrt(252)) if ds_std > 0 else 0

    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    max_dd = float(dd.min()) * 100

    total_return = (equity_curve.iloc[-1] - CAPITAL) / CAPITAL * 100

    avg_days = np.mean([t['days_held'] for t in trades])
    avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
    avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0

    return {
        'total_trades': n,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(float(win_rate), 4),
        'profit_factor': round(float(profit_factor), 3),
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'total_pnl': round(float(total_pnl), 2),
        'total_return_pct': round(float(total_return), 2),
        'max_drawdown_pct': round(float(max_dd), 2),
        'final_equity': round(float(equity_curve.iloc[-1]), 2),
        'avg_days_held': round(float(avg_days), 1),
        'avg_win': round(float(avg_win), 2),
        'avg_loss': round(float(avg_loss), 2),
    }


def regime_analysis(trades, prices):
    """Stratify trades by VIX regime and SPY trend."""
    import yfinance as yf

    # Get VIX data
    vix_data = yf.download(VIX_TICKER, start='2026-01-01', end='2026-08-23',
                           auto_adjust=True, progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_close = vix_data['Close']
        if isinstance(vix_close, pd.DataFrame):
            vix_close = vix_close.iloc[:, 0]
    else:
        vix_close = vix_data['Close'] if 'Close' in vix_data.columns else pd.Series(20.0)

    # Get SPY data for regime
    spy_data = prices[BENCHMARK] if BENCHMARK in prices.columns else None

    regime_trades = {'low_vix': [], 'mid_vix': [], 'high_vix': []}
    trend_trades = {'bull': [], 'bear': [], 'flat': []}

    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])

        # VIX regime
        vix_val = 20.0
        if len(vix_close) > 0:
            vix_near = vix_close.index[vix_close.index <= entry_date]
            if len(vix_near) > 0:
                vix_val = float(vix_close.loc[vix_near[-1]])

        if vix_val < 16:
            regime_trades['low_vix'].append(t)
        elif vix_val > 25:
            regime_trades['high_vix'].append(t)
        else:
            regime_trades['mid_vix'].append(t)

        # SPY trend (20d return)
        if spy_data is not None:
            spy_near = spy_data.index[spy_data.index <= entry_date]
            if len(spy_near) > 20:
                spy_ret20 = (spy_data.loc[spy_near[-1]] - spy_data.loc[spy_near[-21]]) / spy_data.loc[spy_near[-21]]
                if spy_ret20 > 0.02:
                    trend_trades['bull'].append(t)
                elif spy_ret20 < -0.02:
                    trend_trades['bear'].append(t)
                else:
                    trend_trades['flat'].append(t)

    result = {}
    for label, tlist in {**regime_trades, **trend_trades}.items():
        if len(tlist) == 0:
            result[label] = {'trades': 0, 'win_rate': 0, 'avg_return_pct': 0, 'total_pnl': 0}
        else:
            wins = len([t for t in tlist if t['pnl'] > 0])
            result[label] = {
                'trades': len(tlist),
                'win_rate': round(wins / len(tlist), 4),
                'avg_return_pct': round(np.mean([t['return_pct'] for t in tlist]), 4),
                'total_pnl': round(sum(t['pnl'] for t in tlist), 2),
            }

    return result


def monthly_breakdown(equity_curve):
    """Monthly equity and return breakdown."""
    months = {}
    for date in equity_curve.index:
        m = str(date)[:7]
        months[m] = round(float(equity_curve.loc[date]), 2)

    # Compute monthly returns
    month_keys = sorted(months.keys())
    monthly_returns = {}
    for i, m in enumerate(month_keys):
        if i == 0:
            monthly_returns[m] = round((months[m] - CAPITAL) / CAPITAL * 100, 2)
        else:
            prev_m = month_keys[i-1]
            monthly_returns[m] = round((months[m] - months[prev_m]) / months[prev_m] * 100, 2)

    return {'end_equity': months, 'monthly_return_pct': monthly_returns}


def exit_reason_breakdown(trades):
    """Breakdown by exit reason."""
    reasons = {}
    for t in trades:
        r = t['exit_reason']
        if r not in reasons:
            reasons[r] = {'trades': 0, 'pnl': 0, 'wins': 0}
        reasons[r]['trades'] += 1
        reasons[r]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            reasons[r]['wins'] += 1

    for r in reasons:
        reasons[r]['pnl'] = round(reasons[r]['pnl'], 2)
        reasons[r]['win_rate'] = round(reasons[r]['wins'] / reasons[r]['trades'], 4) if reasons[r]['trades'] > 0 else 0

    return reasons


def etf_breakdown(trades):
    """Breakdown by ETF."""
    etfs = {}
    for t in trades:
        ticker = t['ticker']
        if ticker not in etfs:
            etfs[ticker] = {'trades': 0, 'pnl': 0, 'wins': 0}
        etfs[ticker]['trades'] += 1
        etfs[ticker]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            etfs[ticker]['wins'] += 1

    for e in etfs:
        etfs[e]['pnl'] = round(etfs[e]['pnl'], 2)
        etfs[e]['win_rate'] = round(etfs[e]['wins'] / etfs[e]['trades'], 4) if etfs[e]['trades'] > 0 else 0

    return dict(sorted(etfs.items(), key=lambda x: x[1]['pnl'], reverse=True))


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("="*70)
    print("TRUE OUT-OF-SAMPLE LOCKBOX: Flow Reversal Strategy")
    print(f"Strategy: AVO v25, score 4.36")
    print(f"OOS window: {OOS_START} to {OOS_END}")
    print(f"Capital: ${CAPITAL:,.0f}")
    print("="*70)

    # Build data
    feature_data, prices, volume = build_feature_data()

    results = {}

    # ─── Test 1: MAX_CONCURRENT=2 (strategy default) ───────────────────────
    print("\n" + "─"*70)
    print("TEST 1: MAX_CONCURRENT = 2 (strategy default)")
    print("─"*70)

    eq2, trades2, dpnl2 = run_backtest(feature_data, prices, max_concurrent_override=2)

    if trades2 and len(trades2) > 0:
        metrics2 = compute_metrics(trades2, dpnl2, eq2)
        regime2 = regime_analysis(trades2, prices)
        monthly2 = monthly_breakdown(eq2)
        exit2 = exit_reason_breakdown(trades2)
        etf2 = etf_breakdown(trades2)

        print(f"\n  Total trades:     {metrics2['total_trades']}")
        print(f"  Win rate:         {metrics2['win_rate']:.1%}")
        print(f"  Profit factor:    {metrics2['profit_factor']:.2f}")
        print(f"  Sharpe:           {metrics2['sharpe']:.2f}")
        print(f"  Sortino:          {metrics2['sortino']:.2f}")
        print(f"  Total P&L:        ${metrics2['total_pnl']:,.2f}")
        print(f"  Total return:     {metrics2['total_return_pct']:.1f}%")
        print(f"  Max drawdown:     {metrics2['max_drawdown_pct']:.1f}%")
        print(f"  Final equity:     ${metrics2['final_equity']:,.2f}")
        print(f"  Avg days held:    {metrics2['avg_days_held']:.1f}")
        print(f"  Avg win:          ${metrics2['avg_win']:.2f}")
        print(f"  Avg loss:         ${metrics2['avg_loss']:.2f}")

        print("\n  Regime breakdown:")
        for regime, data in regime2.items():
            if data['trades'] > 0:
                print(f"    {regime}: {data['trades']} trades, WR {data['win_rate']:.1%}, "
                      f"avg ret {data['avg_return_pct']:.2f}%, P&L ${data['total_pnl']:.2f}")

        print("\n  Exit reasons:")
        for reason, data in exit2.items():
            print(f"    {reason}: {data['trades']} trades, WR {data['win_rate']:.1%}, P&L ${data['pnl']:.2f}")

        print("\n  Top ETFs by P&L:")
        for etf, data in list(etf2.items())[:8]:
            print(f"    {etf}: {data['trades']} trades, WR {data['win_rate']:.1%}, P&L ${data['pnl']:.2f}")

        print("\n  Monthly returns:")
        for m, ret in monthly2['monthly_return_pct'].items():
            eq_end = monthly2['end_equity'][m]
            print(f"    {m}: {ret:+.2f}% (equity ${eq_end:,.2f})")

        results['max_concurrent_2'] = {
            'metrics': metrics2,
            'regime': regime2,
            'monthly': monthly2,
            'exit_reasons': exit2,
            'etf_breakdown': etf2,
            'trades': trades2,
        }
    else:
        print("  NO TRADES in 2026 OOS window!")
        results['max_concurrent_2'] = {'metrics': {}, 'trades': []}

    # ─── Test 2: MAX_CONCURRENT=1 ──────────────────────────────────────────
    print("\n" + "─"*70)
    print("TEST 2: MAX_CONCURRENT = 1")
    print("─"*70)

    eq1, trades1, dpnl1 = run_backtest(feature_data, prices, max_concurrent_override=1)

    if trades1 and len(trades1) > 0:
        metrics1 = compute_metrics(trades1, dpnl1, eq1)
        regime1 = regime_analysis(trades1, prices)
        monthly1 = monthly_breakdown(eq1)
        exit1 = exit_reason_breakdown(trades1)
        etf1 = etf_breakdown(trades1)

        print(f"\n  Total trades:     {metrics1['total_trades']}")
        print(f"  Win rate:         {metrics1['win_rate']:.1%}")
        print(f"  Profit factor:    {metrics1['profit_factor']:.2f}")
        print(f"  Sharpe:           {metrics1['sharpe']:.2f}")
        print(f"  Sortino:          {metrics1['sortino']:.2f}")
        print(f"  Total P&L:        ${metrics1['total_pnl']:,.2f}")
        print(f"  Total return:     {metrics1['total_return_pct']:.1f}%")
        print(f"  Max drawdown:     {metrics1['max_drawdown_pct']:.1f}%")
        print(f"  Final equity:     ${metrics1['final_equity']:,.2f}")
        print(f"  Avg days held:    {metrics1['avg_days_held']:.1f}")

        print("\n  Regime breakdown:")
        for regime, data in regime1.items():
            if data['trades'] > 0:
                print(f"    {regime}: {data['trades']} trades, WR {data['win_rate']:.1%}, "
                      f"avg ret {data['avg_return_pct']:.2f}%, P&L ${data['total_pnl']:.2f}")

        print("\n  Monthly returns:")
        for m, ret in monthly1['monthly_return_pct'].items():
            eq_end = monthly1['end_equity'][m]
            print(f"    {m}: {ret:+.2f}% (equity ${eq_end:,.2f})")

        results['max_concurrent_1'] = {
            'metrics': metrics1,
            'regime': regime1,
            'monthly': monthly1,
            'exit_reasons': exit1,
            'etf_breakdown': etf1,
            'trades': trades1,
        }
    else:
        print("  NO TRADES in 2026 OOS window!")
        results['max_concurrent_1'] = {'metrics': {}, 'trades': []}

    # ─── Save results ──────────────────────────────────────────────────────
    output = {
        'strategy': 'flow_reversal',
        'avo_version': 'v25',
        'avo_score': 4.36,
        'oos_window': f'{OOS_START} to {OOS_END}',
        'capital': CAPITAL,
        'train_window': f'{TRAIN_START} to {TRAIN_END}',
        'strategy_params': {
            'volume_lookback': strategy.VOLUME_LOOKBACK,
            'volume_z_threshold': strategy.VOLUME_Z_THRESHOLD,
            'price_dip_threshold': strategy.PRICE_DIP_THRESHOLD,
            'trailing_stop_pct': strategy.TRAILING_STOP_PCT,
            'take_profit_pct': strategy.TAKE_PROFIT_PCT,
            'max_hold_days': strategy.MAX_HOLD_DAYS,
            'underwater_cut_days': strategy.UNDERWATER_CUT_DAYS,
            'underwater_tolerance': strategy.UNDERWATER_TOLERANCE,
            'max_per_trade': strategy.MAX_PER_TRADE,
            'slippage_pct': strategy.SLIPPAGE_PCT,
        },
    }

    for key in ['max_concurrent_2', 'max_concurrent_1']:
        if key in results and results[key].get('metrics'):
            # Remove full trade list from JSON (keep just summary)
            r = results[key].copy()
            r['trade_count'] = len(r.get('trades', []))
            # Keep first 5 trades as sample
            r['sample_trades'] = r.get('trades', [])[:5]
            del r['trades']
            output[key] = r

    output_path = '/home/jupiter/Lvl3Quant/validation/flow_reversal_lockbox_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # ─── Final verdict ─────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("LOCKBOX VERDICT")
    print("="*70)

    for label, key in [("MAX_CONCURRENT=2", "max_concurrent_2"), ("MAX_CONCURRENT=1", "max_concurrent_1")]:
        m = results.get(key, {}).get('metrics', {})
        if m:
            sharpe = m.get('sharpe', 0)
            verdict = "PASS" if sharpe > 0.5 and m.get('profit_factor', 0) > 1.0 else "MARGINAL" if sharpe > 0 else "FAIL"
            print(f"  {label}: Sharpe={sharpe:.2f}, PF={m.get('profit_factor',0):.2f}, "
                  f"WR={m.get('win_rate',0):.1%}, Return={m.get('total_return_pct',0):.1f}% → {verdict}")
