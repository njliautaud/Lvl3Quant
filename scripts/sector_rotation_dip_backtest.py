#!/usr/bin/env python3
"""
Sector Rotation Dip Buying Backtest
====================================
Tests 6 variants of sector-level mean reversion using sector ETFs.
OOT: Jan 2022 - Jul 2026 | Capital: $645 | Max $200/trade, max 3 concurrent

Variants:
  A: Simple Sector Dip (>3% from 20d high, RSI<35, hold 10d)
  B: Dual Signal on Sectors (A + first green after 2+ red days, hold 10d)
  C: Relative Sector Weakness (2 weakest sectors when SPY RSI<40, hold 15d)
  D: Sector Divergence (lag SPY >5% over 20d, hold 10d)
  E: VIX-Filtered Sector Dip (A + VIX>20, hold 10d)
  F: Cross-Sector Confirmation (A + >=3 sectors below 20d SMA, hold 10d)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'
START = '2021-06-01'  # extra lookback for indicators
OOT_START = '2022-01-01'
OOT_END = '2026-07-31'
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0001  # 0.01% each way
N_PERM = 1000
RSI_PERIOD = 14
LOOKBACK_20 = 20

# ── Helpers ─────────────────────────────────────────────────────────────

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def download_data():
    tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]
    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close']
    else:
        close = raw

    # Flatten column names if needed
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = [c[0] if isinstance(c, tuple) else c for c in close.columns]

    close = close.ffill().dropna(how='all')
    return close


def run_backtest(signals_df, prices_df, hold_days, variant_name):
    """
    Generic backtest engine.
    signals_df: DataFrame with same index as prices, columns = sector ETFs, values = True/False for entry signal.
    prices_df: Close prices for sector ETFs.
    hold_days: number of trading days to hold.
    Returns dict of metrics.
    """
    oot_mask = (prices_df.index >= OOT_START) & (prices_df.index <= OOT_END)
    dates = prices_df.index[oot_mask]

    trades = []
    open_positions = []  # list of dicts: {ticker, entry_date, entry_price, exit_date_idx, size_dollars}

    daily_pnl = pd.Series(0.0, index=dates)
    equity_curve = pd.Series(CAPITAL, index=dates)

    capital = CAPITAL

    for i, date in enumerate(dates):
        # Check exits
        new_open = []
        for pos in open_positions:
            days_held = np.busday_count(
                np.datetime64(pos['entry_date'], 'D'),
                np.datetime64(date, 'D')
            )
            if days_held >= hold_days:
                # Exit
                exit_price = prices_df.loc[date, pos['ticker']]
                if pd.isna(exit_price):
                    new_open.append(pos)
                    continue
                slippage = exit_price * SLIPPAGE_PCT
                exit_price_adj = exit_price - slippage  # selling
                shares = pos['shares']
                pnl = (exit_price_adj - pos['entry_price_adj']) * shares
                capital += pos['size_dollars'] + pnl
                daily_pnl.iloc[i] += pnl
                trades.append({
                    'ticker': pos['ticker'],
                    'entry_date': str(pos['entry_date'].date()) if hasattr(pos['entry_date'], 'date') else str(pos['entry_date']),
                    'exit_date': str(date.date()) if hasattr(date, 'date') else str(date),
                    'entry_price': pos['entry_price'],
                    'exit_price': float(exit_price),
                    'pnl': float(pnl),
                    'return_pct': float(pnl / pos['size_dollars'] * 100),
                    'hold_days': int(days_held)
                })
            else:
                # Mark-to-market
                curr_price = prices_df.loc[date, pos['ticker']]
                if not pd.isna(curr_price):
                    prev_idx = max(0, i - 1)
                    prev_date = dates[prev_idx]
                    prev_price = prices_df.loc[prev_date, pos['ticker']]
                    if not pd.isna(prev_price) and i > 0:
                        daily_pnl.iloc[i] += (curr_price - prev_price) * pos['shares']
                new_open.append(pos)
        open_positions = new_open

        # Check entries
        if len(open_positions) < MAX_CONCURRENT:
            signal_cols = [c for c in signals_df.columns if c in SECTOR_ETFS]
            for ticker in signal_cols:
                if len(open_positions) >= MAX_CONCURRENT:
                    break
                if date not in signals_df.index:
                    continue
                if signals_df.loc[date, ticker]:
                    # Check not already in position
                    if any(p['ticker'] == ticker for p in open_positions):
                        continue
                    price = prices_df.loc[date, ticker]
                    if pd.isna(price) or price <= 0:
                        continue
                    size = min(MAX_PER_TRADE, capital * 0.95)  # keep 5% buffer
                    if size < 10:
                        continue
                    slippage = price * SLIPPAGE_PCT
                    entry_price_adj = price + slippage  # buying
                    shares = int(size / entry_price_adj)
                    if shares < 1:
                        continue
                    actual_cost = shares * entry_price_adj
                    capital -= actual_cost
                    open_positions.append({
                        'ticker': ticker,
                        'entry_date': date,
                        'entry_price': float(price),
                        'entry_price_adj': entry_price_adj,
                        'shares': shares,
                        'size_dollars': actual_cost
                    })

        if i > 0:
            equity_curve.iloc[i] = equity_curve.iloc[i-1] + daily_pnl.iloc[i]
        else:
            equity_curve.iloc[i] = CAPITAL + daily_pnl.iloc[i]

    # Force close remaining
    last_date = dates[-1]
    for pos in open_positions:
        exit_price = prices_df.loc[last_date, pos['ticker']]
        if pd.isna(exit_price):
            continue
        slippage = exit_price * SLIPPAGE_PCT
        exit_price_adj = exit_price - slippage
        pnl = (exit_price_adj - pos['entry_price_adj']) * pos['shares']
        trades.append({
            'ticker': pos['ticker'],
            'entry_date': str(pos['entry_date'].date()),
            'exit_date': str(last_date.date()),
            'pnl': float(pnl),
            'return_pct': float(pnl / pos['size_dollars'] * 100),
            'hold_days': int(np.busday_count(np.datetime64(pos['entry_date'], 'D'), np.datetime64(last_date, 'D')))
        })

    return compute_metrics(trades, daily_pnl, equity_curve, prices_df, variant_name)


def compute_metrics(trades, daily_pnl, equity_curve, prices_df, variant_name):
    n_trades = len(trades)
    if n_trades == 0:
        return {
            'variant': variant_name, 'trades': 0, 'sharpe': 0, 'sortino': 0,
            'win_rate': 0, 'profit_factor': 0, 'max_drawdown_pct': 0,
            'total_return_pct': 0, 'bull_sharpe': 0, 'bear_sharpe': 0,
            'regime_gap': 999, 'perm_p_value': 1.0, 'passed_gates': 0,
            'gates': {}, 'trade_details': []
        }

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(wins) / n_trades

    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.001
    profit_factor = gross_profit / gross_loss

    total_pnl = sum(t['pnl'] for t in trades)
    total_return_pct = (total_pnl / CAPITAL) * 100

    # Daily returns for Sharpe/Sortino
    daily_ret = daily_pnl / CAPITAL
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0).fillna(0)

    ann_factor = np.sqrt(252)
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std()
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0

    downside = daily_ret[daily_ret < 0].std()
    sortino = (mean_ret / downside * ann_factor) if downside > 0 else 0

    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min()) * 100

    # Regime analysis (Bull: SPY > 200-SMA, Bear: SPY < 200-SMA)
    spy = prices_df[BENCHMARK].reindex(daily_pnl.index).ffill()
    spy_sma200 = spy.rolling(200).mean()
    bull_mask = spy > spy_sma200
    bear_mask = ~bull_mask

    bull_ret = daily_ret[bull_mask]
    bear_ret = daily_ret[bear_mask]

    bull_sharpe = (bull_ret.mean() / bull_ret.std() * ann_factor) if len(bull_ret) > 20 and bull_ret.std() > 0 else 0
    bear_sharpe = (bear_ret.mean() / bear_ret.std() * ann_factor) if len(bear_ret) > 20 and bear_ret.std() > 0 else 0

    max_regime = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_regime if max_regime > 0 else 999

    # Permutation test: randomly flip trade return signs to test if edge is real
    trade_rets = np.array([t['pnl'] for t in trades])
    actual_mean = trade_rets.mean()
    perm_means = []
    for _ in range(N_PERM):
        signs = np.random.choice([-1, 1], size=len(trade_rets))
        perm_means.append((trade_rets * signs).mean())
    perm_p = np.mean([pm >= actual_mean for pm in perm_means])

    # 5-Gate Validation
    gates = {
        'sharpe_gt_0.5': sharpe > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'mdd_gt_neg50': max_dd > -50,
        'trades_gte_20': n_trades >= 20
    }
    passed = sum(gates.values())

    return {
        'variant': variant_name,
        'trades': n_trades,
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'win_rate': round(float(win_rate), 4),
        'profit_factor': round(float(profit_factor), 3),
        'max_drawdown_pct': round(float(max_dd), 2),
        'total_return_pct': round(float(total_return_pct), 2),
        'total_pnl': round(float(total_pnl), 2),
        'bull_sharpe': round(float(bull_sharpe), 3),
        'bear_sharpe': round(float(bear_sharpe), 3),
        'regime_gap': round(float(regime_gap), 3),
        'perm_p_value': round(float(perm_p), 4),
        'passed_gates': passed,
        'gates': gates,
        'avg_trade_pnl': round(float(total_pnl / n_trades), 2),
        'avg_hold_days': round(float(np.mean([t['hold_days'] for t in trades])), 1),
    }


# ── Signal Generators ──────────────────────────────────────────────────

def variant_a_signals(prices, spy, vix):
    """Simple Sector Dip: >3% from 20d high + RSI<35"""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in prices.columns:
            continue
        p = prices[etf]
        high_20 = p.rolling(LOOKBACK_20).max()
        dip = (p - high_20) / high_20
        rsi = compute_rsi(p, RSI_PERIOD)
        signals[etf] = (dip < -0.03) & (rsi < 35)
    return signals


def variant_b_signals(prices, spy, vix):
    """Dual Signal: A + first green after 2+ red days"""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)
    for etf in SECTOR_ETFS:
        if etf not in prices.columns:
            continue
        p = prices[etf]
        high_20 = p.rolling(LOOKBACK_20).max()
        dip = (p - high_20) / high_20
        rsi = compute_rsi(p, RSI_PERIOD)

        daily_ret = p.pct_change()
        red_day = daily_ret < 0
        green_day = daily_ret > 0

        # Count consecutive red days
        consec_red = pd.Series(0, index=p.index)
        for i in range(1, len(p)):
            if red_day.iloc[i]:
                consec_red.iloc[i] = consec_red.iloc[i-1] + 1
            else:
                consec_red.iloc[i] = 0

        # First green after 2+ red
        prev_consec_red = consec_red.shift(1)
        first_green_after_red = green_day & (prev_consec_red >= 2)

        signals[etf] = (dip < -0.03) & (rsi < 35) & first_green_after_red
    return signals


def variant_c_signals(prices, spy, vix):
    """Relative Sector Weakness: 2 weakest sectors when SPY RSI<40"""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)
    spy_rsi = compute_rsi(spy, RSI_PERIOD)

    available_etfs = [e for e in SECTOR_ETFS if e in prices.columns]

    # 20-day returns for each sector
    ret_20 = pd.DataFrame(index=prices.index, columns=available_etfs)
    for etf in available_etfs:
        ret_20[etf] = prices[etf].pct_change(LOOKBACK_20)

    for date in prices.index:
        if spy_rsi.get(date, 50) >= 40:
            continue
        rets = ret_20.loc[date].dropna()
        if len(rets) < 3:
            continue
        weakest_2 = rets.nsmallest(2).index
        for etf in weakest_2:
            signals.loc[date, etf] = True

    return signals


def variant_d_signals(prices, spy, vix):
    """Sector Divergence: lag SPY by >5% over 20 days"""
    signals = pd.DataFrame(False, index=prices.index, columns=SECTOR_ETFS)
    spy_ret_20 = spy.pct_change(LOOKBACK_20)

    for etf in SECTOR_ETFS:
        if etf not in prices.columns:
            continue
        etf_ret_20 = prices[etf].pct_change(LOOKBACK_20)
        lag = etf_ret_20 - spy_ret_20
        signals[etf] = lag < -0.05
    return signals


def variant_e_signals(prices, spy, vix):
    """VIX-Filtered Sector Dip: A + VIX>20"""
    base = variant_a_signals(prices, spy, vix)
    vix_high = vix > 20
    vix_high = vix_high.reindex(base.index).ffill().fillna(False)
    for etf in SECTOR_ETFS:
        if etf in base.columns:
            base[etf] = base[etf] & vix_high
    return base


def variant_f_signals(prices, spy, vix):
    """Cross-Sector Confirmation: A + >=3 sectors below 20d SMA"""
    base_signals = variant_a_signals(prices, spy, vix)

    available_etfs = [e for e in SECTOR_ETFS if e in prices.columns]

    # Count sectors below 20d SMA
    below_sma_count = pd.Series(0, index=prices.index)
    for etf in available_etfs:
        sma20 = prices[etf].rolling(LOOKBACK_20).mean()
        below_sma_count += (prices[etf] < sma20).astype(int)

    broad_weakness = below_sma_count >= 3

    for etf in SECTOR_ETFS:
        if etf in base_signals.columns:
            base_signals[etf] = base_signals[etf] & broad_weakness
    return base_signals


# ── Main ────────────────────────────────────────────────────────────────

def main():
    prices = download_data()
    print(f"Data shape: {prices.shape}, date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    spy = prices[BENCHMARK]
    vix = prices[VIX_TICKER] if VIX_TICKER in prices.columns else prices.get('^VIX', pd.Series(dtype=float))

    # If VIX column name differs
    vix_cols = [c for c in prices.columns if 'VIX' in str(c).upper()]
    if len(vix_cols) > 0 and vix.empty:
        vix = prices[vix_cols[0]]

    sector_prices = prices[[c for c in SECTOR_ETFS if c in prices.columns]]

    variants = {
        'A_Simple_Sector_Dip': (variant_a_signals, 10),
        'B_Dual_Signal_Sectors': (variant_b_signals, 10),
        'C_Relative_Weakness': (variant_c_signals, 15),
        'D_Sector_Divergence': (variant_d_signals, 10),
        'E_VIX_Filtered_Dip': (variant_e_signals, 10),
        'F_Cross_Sector_Confirm': (variant_f_signals, 10),
    }

    results = {}
    for name, (sig_func, hold_days) in variants.items():
        print(f"\n{'='*60}")
        print(f"Running {name} (hold={hold_days}d)...")
        signals = sig_func(prices, spy, vix)

        # Count signals in OOT
        oot_signals = signals[(signals.index >= OOT_START) & (signals.index <= OOT_END)]
        total_sigs = oot_signals.sum().sum()
        print(f"  Total entry signals in OOT: {total_sigs}")

        result = run_backtest(signals, prices, hold_days, name)
        results[name] = result

        print(f"  Trades: {result['trades']}")
        print(f"  Sharpe: {result['sharpe']}, Sortino: {result['sortino']}")
        print(f"  WR: {result['win_rate']:.1%}, PF: {result['profit_factor']}")
        print(f"  MDD: {result['max_drawdown_pct']:.1f}%, Return: {result['total_return_pct']:.1f}%")
        print(f"  Bull Sharpe: {result['bull_sharpe']}, Bear Sharpe: {result['bear_sharpe']}, Gap: {result['regime_gap']}")
        print(f"  Perm p-value: {result['perm_p_value']}")
        print(f"  Gates passed: {result['passed_gates']}/5 {result['gates']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — SECTOR ROTATION DIP BUYING")
    print(f"{'='*60}")
    print(f"{'Variant':<30} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD%':>7} {'Ret%':>7} {'Gates':>5}")
    print("-" * 90)
    for name, r in results.items():
        print(f"{name:<30} {r['trades']:>6} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} {r['max_drawdown_pct']:>6.1f}% {r['total_return_pct']:>6.1f}% {r['passed_gates']:>3}/5")

    # Find best
    valid = {k: v for k, v in results.items() if v['trades'] >= 10}
    if valid:
        best = max(valid, key=lambda k: valid[k]['sharpe'])
        print(f"\nBest variant by Sharpe: {best} (Sharpe={results[best]['sharpe']})")

    # Save results
    output = {
        'metadata': {
            'backtest': 'Sector Rotation Dip Buying',
            'oot_period': f'{OOT_START} to {OOT_END}',
            'starting_capital': CAPITAL,
            'max_per_trade': MAX_PER_TRADE,
            'max_concurrent': MAX_CONCURRENT,
            'slippage_pct': SLIPPAGE_PCT,
            'sector_etfs': SECTOR_ETFS,
            'run_date': datetime.now().isoformat(),
            'permutation_iterations': N_PERM
        },
        'results': results
    }

    out_path = '/home/jupiter/Lvl3Quant/data/sector_rotation_dip_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
