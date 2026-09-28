#!/usr/bin/env python3
"""
TRUE Out-of-Sample Lockbox Validation: Macro Regime Rotation (AVO v19, score 8.00)
==================================================================================
Tests the AVO-evolved macro regime rotation strategy on 2026 data (Jan 2 - Aug 22)
that was NEVER seen during evolution. The strategy evolved on 2022-2025
walk-forward folds (8 folds, 295 trades, geomean Sharpe 4.02, regime gap 0.012).

Strategy: Buy defensive sector ETFs (XLU, XLP, XLV, XLRE) on dips, with
adaptive dip thresholds based on VIX regime, SPY momentum, and fear signals.

Uses yfinance for real market data. No synthetic data.
"""

import sys
import os
import json
import importlib.util
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
STRATEGY_PATH = '/home/jupiter/teleclaude-main/runs/macro_regime_rotation-20260824-001917/work/strategy.py'
RESULTS_PATH = '/home/jupiter/Lvl3Quant/validation/macro_regime_lockbox_results.json'

SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLRE', 'XLB']
TRADEABLE = ['XLU', 'XLP', 'XLV', 'XLRE']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'

OOS_START = '2026-01-02'
OOS_END = '2026-08-22'
WARMUP_START = '2025-06-01'  # enough for 30d trend + 20d SPY mom + 5d dip lookbacks

CAPITAL = 10000.0

# Evolution reference metrics
EVO_SCORE = 8.00
EVO_GEOMEAN_SHARPE = 4.02
EVO_REGIME_GAP = 0.012

# ---------------------------------------------------------------------------
# Load strategy module
# ---------------------------------------------------------------------------
def load_strategy():
    spec = importlib.util.spec_from_file_location("strategy", STRATEGY_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Build macro dial DataFrame (simplified -- from VIX + yield curve proxies)
# ---------------------------------------------------------------------------
def build_macro_dial(prices, vix_series):
    """
    Build a simplified macro_dial DataFrame with fear signals.
    Uses VIX term structure proxy and yield curve proxy from market data.
    The strategy checks for: gate_vix_term_inverted, gate_yc_inverted, gate_dxy_strong
    and a risk_dial column.

    For true OOS we build these from available market data.
    """
    rows = []
    for date in prices.index:
        vix_val = vix_series.get(date, 20.0)
        if pd.isna(vix_val):
            vix_val = 20.0

        # Simple risk dial: normalize VIX to 0-1 scale (10=low risk, 40=high risk)
        dial = np.clip((vix_val - 10) / 30, 0, 1)

        # VIX term inversion proxy: VIX > 30 suggests term structure likely inverted
        vix_term_inv = vix_val > 30

        # Yield curve inversion proxy: not available from yfinance easily,
        # set to False (conservative -- fewer fear signals = harder test)
        yc_inv = False

        # DXY strong proxy: not available, set to False (conservative)
        dxy_strong = False

        rows.append({
            'date': date,
            'risk_dial': dial,
            'gate_vix_term_inverted': vix_term_inv,
            'gate_yc_inverted': yc_inv,
            'gate_dxy_strong': dxy_strong,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------
def run_simulation(strategy_mod, prices, spy_series, vix_series, macro_dial):
    """
    Run the macro regime rotation strategy on 2026 OOS data.
    The strategy's generate_signals() takes raw price DataFrames.
    """
    max_per_trade = getattr(strategy_mod, 'MAX_PER_TRADE', 2000.0)
    max_concurrent = getattr(strategy_mod, 'MAX_CONCURRENT', 1)
    slippage_pct = getattr(strategy_mod, 'SLIPPAGE_PCT', 0.000025)

    # Generate signals on all data (strategy handles lookback internally)
    print("  Generating signals...")
    signals_df = strategy_mod.generate_signals(
        prices, spy_series, vix_series,
        insider_data=None, cross_asset=None,
        macro_dial=macro_dial, sector_rotation=None
    )
    print(f"  Signal DataFrame shape: {signals_df.shape}")

    # Count signal days in OOS period
    oos_mask = (signals_df.index >= OOS_START) & (signals_df.index <= OOS_END)
    oos_signals = signals_df[oos_mask]
    total_signals = oos_signals.sum().sum()
    print(f"  Total entry signals in OOS period: {int(total_signals)}")

    # Get OOS trading dates
    oos_dates = prices.index[(prices.index >= OOS_START) & (prices.index <= OOS_END)]

    capital = CAPITAL
    trades = []
    open_positions = []
    daily_pnl = pd.Series(0.0, index=oos_dates)
    equity_curve = pd.Series(CAPITAL, index=oos_dates)
    peak_equity = CAPITAL

    for i, date in enumerate(oos_dates):
        curr_equity = equity_curve.iloc[max(0, i - 1)] if i > 0 else CAPITAL
        portfolio_dd = (curr_equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

        # -- Process exits first --
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

            # Update HWM for strategy's trailing stop
            hwm = pos.get('high_water_mark', pos['entry_price_adj'])
            if curr_price > hwm:
                pos['high_water_mark'] = curr_price
                pos['hwm'] = curr_price

            # Build position dict that matches what should_exit expects
            pos_for_exit = {
                'entry_date': pos['entry_date'],
                'entry_price': pos['entry_price_adj'],
                'entry_price_adj': pos['entry_price_adj'],
                'hwm': pos.get('high_water_mark', pos['entry_price_adj']),
                'high_water': pos.get('high_water_mark', pos['entry_price_adj']),
            }

            do_exit = strategy_mod.should_exit(pos_for_exit, curr_price, date, portfolio_dd)

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

                # Determine exit reason
                tp = getattr(strategy_mod, 'TAKE_PROFIT_PCT', 0.035)
                mh = getattr(strategy_mod, 'MAX_HOLD_DAYS', 3)
                ue = getattr(strategy_mod, 'UNDERWATER_EXIT_DAYS', 1)
                ts = getattr(strategy_mod, 'TRAILING_STOP_PCT', -0.02)
                hwm_val = pos.get('high_water_mark', entry_p)
                dd_from_high = (curr_price - hwm_val) / hwm_val if hwm_val > 0 else 0

                if pnl_pct >= tp:
                    exit_reason = 'take_profit'
                elif days_held >= mh:
                    exit_reason = 'max_hold'
                elif days_held >= ue and pnl_pct < 0:
                    exit_reason = 'underwater_exit'
                elif dd_from_high <= ts:
                    exit_reason = 'trailing_stop'
                else:
                    exit_reason = 'other'

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
                # Mark-to-market
                if i > 0:
                    prev_date = oos_dates[i - 1]
                    prev_price = prices.loc[prev_date, ticker]
                    if not pd.isna(prev_price):
                        daily_pnl.iloc[i] += (curr_price - prev_price) * pos['shares']
                new_open.append(pos)

        open_positions = new_open

        # -- Check for new entry signals --
        if date in signals_df.index and len(open_positions) < max_concurrent:
            for etf in TRADEABLE:
                if etf not in signals_df.columns:
                    continue
                if not signals_df.loc[date, etf]:
                    continue
                if len(open_positions) >= max_concurrent:
                    break
                if any(p['ticker'] == etf for p in open_positions):
                    continue
                if etf not in prices.columns:
                    continue

                # Execute NEXT day
                next_idx = i + 1
                if next_idx >= len(oos_dates):
                    continue
                next_date = oos_dates[next_idx]
                price = prices.loc[next_date, etf]
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
                    'ticker': etf,
                    'entry_date': next_date,
                    'entry_price_adj': entry_price,
                    'high_water_mark': entry_price,
                    'hwm': entry_price,
                    'shares': shares,
                    'size_dollars': actual_cost,
                })

        # -- Update equity curve --
        if i > 0:
            equity_curve.iloc[i] = equity_curve.iloc[i - 1] + daily_pnl.iloc[i]
        else:
            equity_curve.iloc[i] = CAPITAL + daily_pnl.iloc[i]
        peak_equity = max(peak_equity, equity_curve.iloc[i])

    # -- Force close remaining positions --
    if len(open_positions) > 0:
        last_date = oos_dates[-1]
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

    return trades, daily_pnl, equity_curve


# ---------------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades, daily_pnl, equity_curve, label=""):
    n = len(trades)
    result = {'label': label, 'total_trades': n}

    if n == 0:
        result.update({
            'win_rate': 0, 'profit_factor': 0, 'sharpe': 0, 'sortino': 0,
            'total_pnl': 0, 'total_return_pct': 0, 'max_drawdown_pct': 0,
            'avg_days_held': 0, 'avg_return_pct': 0,
        })
        return result

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    total_pnl = sum(t['pnl'] for t in trades)
    win_rate = len(wins) / n
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    pf = gross_profit / gross_loss

    daily_ret = daily_pnl / CAPITAL
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)
    ann = np.sqrt(252)
    mean_r = daily_ret.mean()
    std_r = daily_ret.std()
    sharpe = float(mean_r / std_r * ann) if std_r > 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    ds_std = downside.std() if len(downside) > 5 else std_r
    sortino = float(mean_r / ds_std * ann) if ds_std > 0 else 0.0

    running_max = equity_curve.cummax()
    dd = (equity_curve - running_max) / running_max
    max_dd = float(dd.min()) * 100

    final_equity = equity_curve.iloc[-1]
    total_return = (final_equity - CAPITAL) / CAPITAL * 100

    avg_days = np.mean([t['days_held'] for t in trades])
    avg_ret = np.mean([t['return_pct'] for t in trades])

    result.update({
        'win_rate': round(win_rate, 4),
        'profit_factor': round(pf, 3),
        'sharpe': round(sharpe, 4),
        'sortino': round(sortino, 4),
        'total_pnl': round(total_pnl, 2),
        'total_return_pct': round(total_return, 2),
        'max_drawdown_pct': round(max_dd, 2),
        'final_equity': round(float(final_equity), 2),
        'avg_days_held': round(float(avg_days), 1),
        'avg_return_pct': round(float(avg_ret), 4),
        'wins': len(wins),
        'losses': len(losses),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
    })
    return result


def regime_analysis(daily_pnl, prices, vix_series):
    """Stratify performance by VIX regime and SPY direction."""
    results = {}
    spy = prices[BENCHMARK]
    spy_ret = spy.pct_change()
    vix_aligned = vix_series.reindex(daily_pnl.index).ffill().fillna(20.0)
    daily_ret = daily_pnl / CAPITAL
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)

    # By VIX regime
    for regime, label in [('low', 'VIX < 16'), ('mid', 'VIX 16-25'), ('high', 'VIX > 25')]:
        if regime == 'low':
            mask = vix_aligned < 16
        elif regime == 'mid':
            mask = (vix_aligned >= 16) & (vix_aligned <= 25)
        else:
            mask = vix_aligned > 25

        regime_rets = daily_ret[mask]
        n_days = len(regime_rets)
        if n_days < 5 or regime_rets.std() == 0:
            results[f'vix_{regime}'] = {'label': label, 'days': n_days, 'sharpe': 0.0}
        else:
            s = float(regime_rets.mean() / regime_rets.std() * np.sqrt(252))
            results[f'vix_{regime}'] = {'label': label, 'days': n_days, 'sharpe': round(s, 4)}

    # By SPY direction
    spy_daily = spy_ret.reindex(daily_pnl.index).fillna(0)
    for direction, label in [('green', 'SPY up days'), ('red', 'SPY down days'), ('flat', 'SPY flat')]:
        if direction == 'green':
            mask = spy_daily > 0.001
        elif direction == 'red':
            mask = spy_daily < -0.001
        else:
            mask = (spy_daily >= -0.001) & (spy_daily <= 0.001)

        dir_rets = daily_ret[mask]
        n_days = len(dir_rets)
        if n_days < 5 or dir_rets.std() == 0:
            results[f'spy_{direction}'] = {'label': label, 'days': n_days, 'sharpe': 0.0}
        else:
            s = float(dir_rets.mean() / dir_rets.std() * np.sqrt(252))
            results[f'spy_{direction}'] = {'label': label, 'days': n_days, 'sharpe': round(s, 4)}

    # Compute regime gap (|green - red| / max)
    green_s = results.get('spy_green', {}).get('sharpe', 0)
    red_s = results.get('spy_red', {}).get('sharpe', 0)
    denom = max(abs(green_s), abs(red_s), 0.001)
    results['regime_gap'] = round(abs(green_s - red_s) / denom, 4)

    return results


def sector_breakdown(trades):
    result = {}
    for etf in sorted(set(t['ticker'] for t in trades)):
        etf_trades = [t for t in trades if t['ticker'] == etf]
        etf_pnl = sum(t['pnl'] for t in etf_trades)
        etf_wins = len([t for t in etf_trades if t['pnl'] > 0])
        result[etf] = {
            'trades': len(etf_trades),
            'wins': etf_wins,
            'pnl': round(etf_pnl, 2),
            'win_rate': round(etf_wins / len(etf_trades), 3) if etf_trades else 0,
        }
    return result


def exit_breakdown(trades):
    result = {}
    for reason in sorted(set(t['exit_reason'] for t in trades)):
        r_trades = [t for t in trades if t['exit_reason'] == reason]
        r_pnl = sum(t['pnl'] for t in r_trades)
        result[reason] = {
            'count': len(r_trades),
            'pnl': round(r_pnl, 2),
            'avg_return_pct': round(np.mean([t['return_pct'] for t in r_trades]), 4),
        }
    return result


def monthly_equity(equity_curve):
    result = {}
    for month in sorted(set(str(d)[:7] for d in equity_curve.index)):
        month_mask = [str(d)[:7] == month for d in equity_curve.index]
        month_vals = equity_curve[month_mask]
        if len(month_vals) > 0:
            result[month] = round(float(month_vals.iloc[-1]), 2)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TRUE OUT-OF-SAMPLE LOCKBOX: Macro Regime Rotation (AVO v19, score 8.00)")
    print("=" * 70)
    print(f"OOS period: {OOS_START} to {OOS_END}")
    print(f"Capital: ${CAPITAL:,.0f}")
    print(f"Evolution metrics: score={EVO_SCORE}, geomean Sharpe={EVO_GEOMEAN_SHARPE}, regime gap={EVO_REGIME_GAP}")
    print()

    # Load strategy
    strategy_mod = load_strategy()
    print(f"Strategy loaded: MAX_CONCURRENT={strategy_mod.MAX_CONCURRENT}, "
          f"SLIPPAGE={strategy_mod.SLIPPAGE_PCT}, MAX_HOLD={strategy_mod.MAX_HOLD_DAYS}d, "
          f"TP={strategy_mod.TAKE_PROFIT_PCT}, TS={strategy_mod.TRAILING_STOP_PCT}")
    print(f"Tradeable sectors: {strategy_mod.TRADEABLE_SECTORS}")
    print(f"Dip thresholds: normal={strategy_mod.DIP_THRESHOLD}, "
          f"highvol={strategy_mod.DIP_THRESHOLD_HIGHVOL}, "
          f"spystrong={strategy_mod.DIP_THRESHOLD_SPYSTRONG}, "
          f"fear={strategy_mod.DIP_THRESHOLD_TERM_INV}")

    # Download prices
    print("\nDownloading price data from yfinance...")
    import yfinance as yf
    tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]
    raw = yf.download(tickers, start=WARMUP_START, end='2026-08-23',
                      auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw['Close']
    else:
        prices = raw

    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = [c[0] if isinstance(c, tuple) else c for c in prices.columns]

    prices = prices.ffill().dropna(how='all')
    print(f"Prices: {prices.index.min().strftime('%Y-%m-%d')} to {prices.index.max().strftime('%Y-%m-%d')}, {len(prices)} days")

    # Extract VIX and SPY series
    vix_cols = [c for c in prices.columns if 'VIX' in str(c).upper()]
    if vix_cols:
        vix_ticker_col = vix_cols[0]
        vix = prices[vix_ticker_col]
        # Drop VIX from prices (not a tradeable)
        prices_clean = prices.drop(columns=[vix_ticker_col])
    else:
        vix = pd.Series(20.0, index=prices.index)
        prices_clean = prices

    spy = prices_clean[BENCHMARK]

    # Build macro dial
    print("Building macro dial from VIX data...")
    macro_dial = build_macro_dial(prices_clean, vix)
    print(f"Macro dial rows: {len(macro_dial)}")

    # Report VIX stats in OOS period
    oos_vix = vix[(vix.index >= OOS_START) & (vix.index <= OOS_END)]
    print(f"\nVIX in OOS period: mean={oos_vix.mean():.1f}, min={oos_vix.min():.1f}, max={oos_vix.max():.1f}")
    vix_high_days = (oos_vix > 25).sum()
    print(f"VIX > 25 days: {vix_high_days} / {len(oos_vix)}")

    # Run simulation
    print(f"\n{'='*70}")
    print("Running OOS simulation...")
    print(f"{'='*70}")

    trades, dpnl, eq = run_simulation(strategy_mod, prices_clean, spy, vix, macro_dial)
    metrics = compute_metrics(trades, dpnl, eq, "Macro Regime Rotation OOS 2026")

    print(f"\n  Total trades:    {metrics['total_trades']}")
    print(f"  Win rate:        {metrics['win_rate']:.1%}")
    print(f"  Profit factor:   {metrics['profit_factor']:.2f}")
    print(f"  Sharpe:          {metrics['sharpe']:.2f}")
    print(f"  Sortino:         {metrics['sortino']:.2f}")
    print(f"  Total P&L:       ${metrics['total_pnl']:,.2f}")
    print(f"  Total return:    {metrics['total_return_pct']:.1f}%")
    print(f"  Max drawdown:    {metrics['max_drawdown_pct']:.1f}%")
    print(f"  Avg days held:   {metrics['avg_days_held']:.1f}")
    print(f"  Avg return/trade: {metrics['avg_return_pct']:.2f}%")

    regime = {}
    sector = {}
    exits = {}
    monthly = {}

    if trades:
        regime = regime_analysis(dpnl, prices_clean, vix)
        sector = sector_breakdown(trades)
        exits = exit_breakdown(trades)
        monthly = monthly_equity(eq)

        print("\n  VIX Regime Sharpes:")
        for k, v in regime.items():
            if k.startswith('vix_'):
                print(f"    {v['label']}: Sharpe={v['sharpe']:.2f} ({v['days']} days)")

        print("\n  SPY Direction Sharpes:")
        for k, v in regime.items():
            if k.startswith('spy_'):
                print(f"    {v['label']}: Sharpe={v['sharpe']:.2f} ({v['days']} days)")

        print(f"\n  OOS Regime Gap: {regime.get('regime_gap', 'N/A')}")
        print(f"  (Evolution regime gap was: {EVO_REGIME_GAP})")

        print("\n  By Sector:")
        for etf, info in sector.items():
            print(f"    {etf}: {info['trades']} trades, WR {info['win_rate']:.0%}, P&L ${info['pnl']:.2f}")

        print("\n  By Exit Reason:")
        for reason, info in exits.items():
            print(f"    {reason}: {info['count']} trades, P&L ${info['pnl']:.2f}, avg {info['avg_return_pct']:.2f}%")

        print("\n  Monthly Equity:")
        for month, eq_val in monthly.items():
            print(f"    {month}: ${eq_val:,.2f}")

        # Print first 10 trades for inspection
        print("\n  Sample trades (first 10):")
        for t in trades[:10]:
            print(f"    {t['entry_date']} -> {t['exit_date']} | {t['ticker']} | "
                  f"${t['entry_price']:.2f}->${t['exit_price']:.2f} | "
                  f"P&L ${t['pnl']:.2f} ({t['return_pct']:.2f}%) | {t['exit_reason']}")

    # ---- Evolution vs OOS comparison ----
    print(f"\n{'='*70}")
    print("EVOLUTION vs OOS COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'Evolution':<15} {'OOS 2026':<15} {'Ratio':<10}")
    print(f"  {'-'*65}")

    oos_sharpe = metrics['sharpe']
    sharpe_ratio = oos_sharpe / EVO_GEOMEAN_SHARPE if EVO_GEOMEAN_SHARPE != 0 else 0
    print(f"  {'Sharpe':<25} {EVO_GEOMEAN_SHARPE:<15.2f} {oos_sharpe:<15.2f} {sharpe_ratio:<10.2f}")

    oos_regime_gap = regime.get('regime_gap', float('nan'))
    print(f"  {'Regime Gap':<25} {EVO_REGIME_GAP:<15.3f} {oos_regime_gap:<15.3f} {'--':<10}")

    evo_trades_per_fold = 295 / 8  # ~37 trades per fold
    oos_months = 7.7  # Jan-Aug
    trades_per_month = metrics['total_trades'] / oos_months if oos_months > 0 else 0
    print(f"  {'Trades':<25} {'295 (8 folds)':<15} {metrics['total_trades']:<15} {'--':<10}")
    print(f"  {'Score':<25} {EVO_SCORE:<15.2f} {'N/A':<15} {'--':<10}")

    # ---- Verdict ----
    print(f"\n{'='*70}")
    print("LOCKBOX VALIDATION VERDICT")
    print(f"{'='*70}")

    pass_criteria = []
    fail_criteria = []

    if metrics['total_trades'] >= 10:
        pass_criteria.append(f"Sufficient trades: {metrics['total_trades']} >= 10")
    else:
        fail_criteria.append(f"Too few trades: {metrics['total_trades']} < 10")

    if metrics['sharpe'] > 0.5:
        pass_criteria.append(f"Sharpe positive: {metrics['sharpe']:.2f} > 0.5")
    else:
        fail_criteria.append(f"Sharpe too low: {metrics['sharpe']:.2f} <= 0.5")

    if metrics['profit_factor'] > 1.0:
        pass_criteria.append(f"Profit factor positive: {metrics['profit_factor']:.2f} > 1.0")
    else:
        fail_criteria.append(f"Profit factor negative: {metrics['profit_factor']:.2f} <= 1.0")

    if metrics['win_rate'] > 0.40:
        pass_criteria.append(f"Win rate acceptable: {metrics['win_rate']:.1%} > 40%")
    else:
        fail_criteria.append(f"Win rate low: {metrics['win_rate']:.1%} <= 40%")

    if metrics['max_drawdown_pct'] > -15:
        pass_criteria.append(f"Max drawdown contained: {metrics['max_drawdown_pct']:.1f}%")
    else:
        fail_criteria.append(f"Max drawdown excessive: {metrics['max_drawdown_pct']:.1f}%")

    # Sharpe decay check: OOS should be at least 25% of evolution
    if sharpe_ratio >= 0.25:
        pass_criteria.append(f"Sharpe decay acceptable: {sharpe_ratio:.0%} of evolution")
    else:
        fail_criteria.append(f"Sharpe decay too steep: {sharpe_ratio:.0%} of evolution (< 25%)")

    verdict = "PASS" if len(fail_criteria) == 0 else "FAIL"
    print(f"\n  VERDICT: {verdict}")
    print()
    for c in pass_criteria:
        print(f"  [PASS] {c}")
    for c in fail_criteria:
        print(f"  [FAIL] {c}")

    # Plain English explanation
    print(f"\n  --- PLAIN ENGLISH ---")
    if verdict == "PASS":
        print(f"  The Macro Regime Rotation strategy PASSED lockbox validation.")
        print(f"  On truly unseen 2026 data (Jan-Aug), it generated {metrics['total_trades']} trades")
        print(f"  with a {metrics['win_rate']:.0%} win rate, earning ${metrics['total_pnl']:.2f} on $10K capital")
        print(f"  ({metrics['total_return_pct']:.1f}% return). The Sharpe ratio of {metrics['sharpe']:.2f}")
        print(f"  is {sharpe_ratio:.0%} of the evolution Sharpe ({EVO_GEOMEAN_SHARPE:.2f}),")
        print(f"  which is normal OOS decay. Max drawdown was {metrics['max_drawdown_pct']:.1f}%.")
    else:
        print(f"  The Macro Regime Rotation strategy FAILED lockbox validation.")
        print(f"  While it scored {EVO_SCORE} in evolution with {EVO_GEOMEAN_SHARPE:.2f} Sharpe,")
        print(f"  on unseen 2026 data it achieved only {metrics['sharpe']:.2f} Sharpe")
        print(f"  ({metrics['total_trades']} trades, {metrics['win_rate']:.0%} WR, {metrics['total_return_pct']:.1f}% return).")
        print(f"  This suggests the strategy may have been overfit to training data regimes.")

    # ---- Save results ----
    save_results = {
        'strategy': 'macro_regime_rotation',
        'avo_version': 'v19',
        'avo_score': EVO_SCORE,
        'avo_geomean_sharpe': EVO_GEOMEAN_SHARPE,
        'avo_regime_gap': EVO_REGIME_GAP,
        'oos_period': f'{OOS_START} to {OOS_END}',
        'capital': CAPITAL,
        'verdict': verdict,
        'strategy_params': {
            'TRADEABLE_SECTORS': strategy_mod.TRADEABLE_SECTORS,
            'MAX_CONCURRENT': strategy_mod.MAX_CONCURRENT,
            'SLIPPAGE_PCT': strategy_mod.SLIPPAGE_PCT,
            'MAX_PER_TRADE': strategy_mod.MAX_PER_TRADE,
            'MAX_HOLD_DAYS': strategy_mod.MAX_HOLD_DAYS,
            'TRAILING_STOP_PCT': strategy_mod.TRAILING_STOP_PCT,
            'TAKE_PROFIT_PCT': strategy_mod.TAKE_PROFIT_PCT,
            'DIP_THRESHOLD': strategy_mod.DIP_THRESHOLD,
            'DIP_THRESHOLD_HIGHVOL': strategy_mod.DIP_THRESHOLD_HIGHVOL,
            'DIP_THRESHOLD_SPYSTRONG': strategy_mod.DIP_THRESHOLD_SPYSTRONG,
            'DIP_THRESHOLD_TERM_INV': strategy_mod.DIP_THRESHOLD_TERM_INV,
            'TREND_LOOKBACK': strategy_mod.TREND_LOOKBACK,
            'TREND_MIN': strategy_mod.TREND_MIN,
        },
        'oos_metrics': metrics,
        'regime_analysis': regime,
        'sector_breakdown': sector,
        'exit_breakdown': exits,
        'monthly_equity': monthly,
        'evolution_comparison': {
            'evo_sharpe': EVO_GEOMEAN_SHARPE,
            'oos_sharpe': oos_sharpe,
            'sharpe_ratio': round(sharpe_ratio, 4),
            'evo_regime_gap': EVO_REGIME_GAP,
            'oos_regime_gap': oos_regime_gap,
        },
        'pass_criteria': pass_criteria,
        'fail_criteria': fail_criteria,
        'trades': trades,
        'timestamp': datetime.now().isoformat(),
        'plain_english': (
            f"The strategy {'PASSED' if verdict == 'PASS' else 'FAILED'} lockbox validation. "
            f"On unseen 2026 data (Jan-Aug), it made {metrics['total_trades']} trades "
            f"with {metrics['win_rate']:.0%} win rate, {metrics['sharpe']:.2f} Sharpe, "
            f"{metrics['profit_factor']:.2f} PF, {metrics['total_return_pct']:.1f}% return, "
            f"{metrics['max_drawdown_pct']:.1f}% max drawdown. "
            f"OOS Sharpe is {sharpe_ratio:.0%} of evolution Sharpe ({EVO_GEOMEAN_SHARPE:.2f})."
        ),
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == '__main__':
    main()
