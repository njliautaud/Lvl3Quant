#!/usr/bin/env python3
"""
Volatility Risk Premium Harvesting Backtest
6 variants (A-F) testing structural VIX contango exploitation
with ETFs accessible in a small Robinhood account.

Universe: SVXY, VIXY, UVXY, SPY, TLT, SHY
VIX data: ^VIX, ^VIX3M (or 63-day MA approximation)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── Parameters ───
START = '2022-01-01'
END = '2026-07-28'
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
PERMUTATION_ITERS = 1000
np.random.seed(42)

# ─── Download Data ───
print("Downloading market data...")
tickers = ['SVXY', 'VIXY', 'UVXY', 'SPY', 'TLT', 'SHY', '^VIX', '^VIX3M']
raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

# Extract close prices
close = raw['Close'].copy()
close.columns = [c if isinstance(c, str) else c for c in close.columns]

# Rename VIX columns
rename_map = {}
for col in close.columns:
    if 'VIX3M' in str(col).upper() or col == '^VIX3M':
        rename_map[col] = 'VIX3M'
    elif 'VIX' in str(col).upper() and '3M' not in str(col).upper() and 'VIXY' not in str(col).upper() and 'SVXY' not in str(col).upper() and 'UVXY' not in str(col).upper():
        rename_map[col] = 'VIX'
close.rename(columns=rename_map, inplace=True)

# If VIX3M has too many NaNs, approximate with 63-day MA of VIX
if 'VIX3M' not in close.columns or close['VIX3M'].isna().sum() > len(close) * 0.5:
    print("VIX3M unavailable or sparse — using VIX 63-day MA as proxy")
    close['VIX3M'] = close['VIX'].rolling(63).mean()
    vix3m_approx = True
else:
    vix3m_approx = False

close = close.dropna(subset=['VIX', 'SPY'])
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")
print(f"VIX3M source: {'63-day MA proxy' if vix3m_approx else 'actual ^VIX3M'}")

# Precompute indicators
vix = close['VIX']
vix3m = close['VIX3M']
vix_3mo_ma = vix.rolling(63).mean()
spy_50sma = close['SPY'].rolling(50).mean()
spy_200sma = close['SPY'].rolling(200).mean()
vix_vix3m_ratio = vix / vix3m

# Regime classification: bull = SPY > 200-SMA, bear = SPY < 200-SMA
regime = pd.Series('bull', index=close.index)
regime[close['SPY'] < spy_200sma] = 'bear'


# ─── Backtest Engine ───
def backtest_variant(signals: pd.DataFrame, variant_name: str):
    """
    signals: DataFrame with columns ['date', 'action', 'ticker']
    action: 'buy' or 'sell' (sell = go to cash)
    Returns dict of metrics.
    """
    equity = CAPITAL
    position = None  # (ticker, shares, entry_price)
    equity_curve = []
    trades = []

    for i, date in enumerate(close.index):
        # Get today's signals (may have sell+buy on same day for rotations)
        day_signals = signals[signals['date'] == date]

        price_today = {}
        for t in ['SVXY', 'VIXY', 'UVXY', 'SPY', 'TLT', 'SHY']:
            p = close[t].get(date, np.nan) if t in close.columns else np.nan
            price_today[t] = p

        # Process all signals for today (sell first, then buy)
        for _, sig in day_signals.sort_values('action', ascending=False).iterrows():
            action = sig['action']
            ticker = sig.get('ticker', None)

            if action == 'sell' and position is not None:
                # Close position
                exit_price = price_today.get(position[0], np.nan)
                if not np.isnan(exit_price):
                    exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
                    proceeds = position[1] * exit_price_adj
                    pnl = proceeds - position[1] * position[2]
                    trades.append({
                        'entry_date': position[3],
                        'exit_date': date,
                        'ticker': position[0],
                        'entry_price': position[2],
                        'exit_price': exit_price_adj,
                        'shares': position[1],
                        'pnl': pnl,
                        'return': pnl / (position[1] * position[2])
                    })
                    equity = proceeds
                    position = None

            if action == 'buy' and position is None and ticker is not None:
                # Open position
                entry_price = price_today.get(ticker, np.nan)
                if not np.isnan(entry_price) and entry_price > 0:
                    entry_price_adj = entry_price * (1 + SLIPPAGE_PCT)
                    shares = equity / entry_price_adj
                    position = (ticker, shares, entry_price_adj, date)
                    # equity stays same until exit

        # Mark to market
        if position is not None:
            mtm_price = price_today.get(position[0], np.nan)
            if not np.isnan(mtm_price):
                equity_curve.append(position[1] * mtm_price)
            else:
                equity_curve.append(equity_curve[-1] if equity_curve else equity)
        else:
            equity_curve.append(equity if not equity_curve else equity)
            # Update cash equity tracking
            equity = equity_curve[-1] if position is None else equity

    equity_series = pd.Series(equity_curve, index=close.index)
    return compute_metrics(equity_series, trades, variant_name)


def compute_metrics(equity_series, trades, variant_name):
    """Compute all required metrics."""
    returns = equity_series.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

    # Basic metrics
    total_return = (equity_series.iloc[-1] / CAPITAL) - 1
    n_years = len(equity_series) / 252
    cagr = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    # Sharpe (annualized, risk-free = 0 for simplicity)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    # Max drawdown
    cummax = equity_series.cummax()
    drawdown = (equity_series - cummax) / cummax
    max_dd = drawdown.min()

    # Trade-level metrics
    n_trades = len(trades)
    if n_trades > 0:
        trade_returns = [t['pnl'] for t in trades]
        wins = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        win_rate = len(wins) / n_trades
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss
    else:
        win_rate = 0
        profit_factor = 0

    # Regime analysis
    bull_mask = regime == 'bull'
    bear_mask = regime == 'bear'
    bull_returns = returns[bull_mask.reindex(returns.index, fill_value=False)]
    bear_returns = returns[bear_mask.reindex(returns.index, fill_value=False)]

    bull_sharpe = (bull_returns.mean() / bull_returns.std() * np.sqrt(252)) if len(bull_returns) > 10 and bull_returns.std() > 0 else 0
    bear_sharpe = (bear_returns.mean() / bear_returns.std() * np.sqrt(252)) if len(bear_returns) > 10 and bear_returns.std() > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    # Permutation test
    p_value = permutation_test(trades, equity_series)

    # 5 gates
    gates = {
        'sharpe_gt_0.5': sharpe > 0.5,
        'perm_p_lt_0.05': p_value < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'max_dd_gt_neg50': max_dd > -0.50,
        'n_trades_gte_20': n_trades >= 20
    }
    gates_passed = sum(gates.values())

    verdict = 'PASS' if gates_passed == 5 else f'FAIL ({5 - gates_passed} gates failed)'

    return {
        'variant': variant_name,
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'total_return': round(float(total_return), 4),
        'max_drawdown': round(float(max_dd), 4),
        'n_trades': int(n_trades),
        'profit_factor': round(float(profit_factor), 4),
        'win_rate': round(float(win_rate), 4),
        'cagr': round(float(cagr), 4),
        'final_equity': round(float(equity_series.iloc[-1]), 2),
        'regime_analysis': {
            'bull_sharpe': round(float(bull_sharpe), 4),
            'bear_sharpe': round(float(bear_sharpe), 4),
            'regime_gap': round(float(regime_gap), 4)
        },
        'permutation_p_value': round(float(p_value), 4),
        'gates': gates,
        'gates_passed': f'{gates_passed}/5',
        'verdict': verdict
    }


def permutation_test(trades, equity_series, n_iter=PERMUTATION_ITERS):
    """Permutation test: shuffle trade entry dates across valid trading days.
    Recompute total return for each shuffle. p = fraction with return >= actual."""
    if len(trades) < 5:
        return 1.0

    actual_total_return = (equity_series.iloc[-1] / CAPITAL) - 1

    # Extract trade durations and the asset traded
    trade_info = []
    for t in trades:
        entry_idx = close.index.get_loc(t['entry_date'])
        exit_idx = close.index.get_loc(t['exit_date'])
        duration = exit_idx - entry_idx
        ticker = t['ticker']
        trade_info.append((duration, ticker))

    all_dates = close.index
    n_dates = len(all_dates)

    count_better = 0
    for _ in range(n_iter):
        # Simulate: place trades at random dates with same durations
        shuf_equity = CAPITAL
        for duration, ticker in trade_info:
            max_start = n_dates - duration - 1
            if max_start < 1:
                continue
            start_idx = np.random.randint(0, max_start)
            end_idx = start_idx + duration

            entry_p = close[ticker].iloc[start_idx]
            exit_p = close[ticker].iloc[end_idx]
            if np.isnan(entry_p) or np.isnan(exit_p) or entry_p <= 0:
                continue
            entry_adj = entry_p * (1 + SLIPPAGE_PCT)
            exit_adj = exit_p * (1 - SLIPPAGE_PCT)
            shuf_equity *= (exit_adj / entry_adj)

        shuf_return = (shuf_equity / CAPITAL) - 1
        if shuf_return >= actual_total_return:
            count_better += 1

    return count_better / n_iter


# ─── Generate Signals for Each Variant ───

def variant_a_signals():
    """VIX Contango Harvest: Buy SVXY when VIX < 3mo MA AND VIX < 20. Exit when VIX > 25 or VIX > 3mo MA."""
    signals = []
    in_position = False

    for date in close.index:
        v = vix.get(date, np.nan)
        ma = vix_3mo_ma.get(date, np.nan)
        if np.isnan(v) or np.isnan(ma):
            continue

        if not in_position:
            if v < ma and v < 20:
                signals.append({'date': date, 'action': 'buy', 'ticker': 'SVXY'})
                in_position = True
        else:
            if v > 25 or v > ma:
                signals.append({'date': date, 'action': 'sell', 'ticker': 'SVXY'})
                in_position = False

    # Close any open position at end
    if in_position:
        signals.append({'date': close.index[-1], 'action': 'sell', 'ticker': 'SVXY'})

    return pd.DataFrame(signals)


def variant_b_signals():
    """Vol Crush After Spikes: Buy SVXY after VIX spikes >25 then drops <22. Hold 10 days."""
    signals = []
    in_position = False
    hold_counter = 0
    spike_seen = False

    for date in close.index:
        v = vix.get(date, np.nan)
        if np.isnan(v):
            continue

        if in_position:
            hold_counter += 1
            if hold_counter >= 10:
                signals.append({'date': date, 'action': 'sell', 'ticker': 'SVXY'})
                in_position = False
                hold_counter = 0
                spike_seen = False
        else:
            if v > 25:
                spike_seen = True
            if spike_seen and v < 22:
                signals.append({'date': date, 'action': 'buy', 'ticker': 'SVXY'})
                in_position = True
                hold_counter = 0
                spike_seen = False

    if in_position:
        signals.append({'date': close.index[-1], 'action': 'sell', 'ticker': 'SVXY'})

    return pd.DataFrame(signals)


def variant_c_signals():
    """VIX Term Structure: VIX/VIX3M < 0.85 → SVXY, > 1.0 → VIXY, 0.85-1.0 → cash."""
    signals = []
    in_position = False
    current_ticker = None

    for date in close.index:
        ratio = vix_vix3m_ratio.get(date, np.nan)
        if np.isnan(ratio):
            continue

        if ratio < 0.85:
            target = 'SVXY'
        elif ratio > 1.0:
            target = 'VIXY'
        else:
            target = None  # cash

        if in_position and target != current_ticker:
            signals.append({'date': date, 'action': 'sell', 'ticker': current_ticker})
            in_position = False
            current_ticker = None

        if not in_position and target is not None:
            signals.append({'date': date, 'action': 'buy', 'ticker': target})
            in_position = True
            current_ticker = target

    if in_position:
        signals.append({'date': close.index[-1], 'action': 'sell', 'ticker': current_ticker})

    return pd.DataFrame(signals)


def variant_d_signals():
    """Tail Hedge Rotation: SPY normally, TLT when VIX>25, SHY when VIX>35, back to SPY when VIX<20.
    Uses hysteresis: only switch back to SPY once VIX drops below 20 (not just below 25)."""
    signals = []
    current_ticker = None
    in_position = False
    hedging = False  # True when in TLT or SHY mode

    for date in close.index:
        v = vix.get(date, np.nan)
        if np.isnan(v):
            continue

        if v > 35:
            target = 'SHY'
            hedging = True
        elif v > 25:
            target = 'TLT'
            hedging = True
        elif hedging and v >= 20:
            # Stay in current hedge until VIX drops below 20
            target = current_ticker if current_ticker in ('TLT', 'SHY') else 'TLT'
        else:
            target = 'SPY'
            hedging = False

        if current_ticker != target:
            if in_position:
                signals.append({'date': date, 'action': 'sell', 'ticker': current_ticker})
                in_position = False
            signals.append({'date': date, 'action': 'buy', 'ticker': target})
            in_position = True
            current_ticker = target

    if in_position:
        signals.append({'date': close.index[-1], 'action': 'sell', 'ticker': current_ticker})

    return pd.DataFrame(signals)


def variant_e_signals():
    """Vol Risk Premium + Momentum: SVXY when VIX<20 AND SPY>50-SMA. TLT when VIX>25. Cash otherwise."""
    signals = []
    in_position = False
    current_ticker = None

    for date in close.index:
        v = vix.get(date, np.nan)
        sma = spy_50sma.get(date, np.nan)
        spy_price = close['SPY'].get(date, np.nan)
        if np.isnan(v) or np.isnan(sma) or np.isnan(spy_price):
            continue

        if v < 20 and spy_price > sma:
            target = 'SVXY'
        elif v > 25:
            target = 'TLT'
        else:
            target = None

        if in_position and target != current_ticker:
            signals.append({'date': date, 'action': 'sell', 'ticker': current_ticker})
            in_position = False
            current_ticker = None

        if not in_position and target is not None:
            signals.append({'date': date, 'action': 'buy', 'ticker': target})
            in_position = True
            current_ticker = target

    if in_position:
        signals.append({'date': close.index[-1], 'action': 'sell', 'ticker': current_ticker})

    return pd.DataFrame(signals)


def variant_f_signals(n_trades_target):
    """ADVERSARIAL: Random VIX-based trades on SVXY. Same trade count as best variant."""
    signals = []
    dates = close.index.tolist()

    # Generate random entry dates
    n_pairs = max(n_trades_target, 10)
    available = list(range(63, len(dates) - 12))  # skip warmup and leave room
    if len(available) < n_pairs:
        n_pairs = len(available) // 2

    entry_indices = sorted(np.random.choice(available, size=min(n_pairs, len(available)), replace=False))

    in_position = False
    trade_count = 0
    hold_counter = 0

    for i, date in enumerate(dates):
        if in_position:
            hold_counter += 1
            hold_target = np.random.randint(3, 15)
            if hold_counter >= hold_target:
                signals.append({'date': date, 'action': 'sell', 'ticker': 'SVXY'})
                in_position = False
                trade_count += 1
                if trade_count >= n_pairs:
                    break
        else:
            if i in entry_indices and not in_position:
                signals.append({'date': date, 'action': 'buy', 'ticker': 'SVXY'})
                in_position = True
                hold_counter = 0

    if in_position:
        signals.append({'date': dates[-1], 'action': 'sell', 'ticker': 'SVXY'})

    return pd.DataFrame(signals)


# ─── Run All Variants ───
print("\n" + "="*70)
print("VOLATILITY RISK PREMIUM HARVESTING BACKTEST")
print(f"Capital: ${CAPITAL}  |  Period: {START} to {END}")
print("="*70)

results = {}

# Variant A
print("\n[A] VIX Contango Harvest...")
sig_a = variant_a_signals()
res_a = backtest_variant(sig_a, 'A_VIX_Contango_Harvest')
results['A'] = res_a
print(f"    Sharpe={res_a['sharpe']:.2f}  Return={res_a['total_return']:.1%}  DD={res_a['max_drawdown']:.1%}  Trades={res_a['n_trades']}  {res_a['verdict']}")

# Variant B
print("[B] Vol Crush After Spikes...")
sig_b = variant_b_signals()
res_b = backtest_variant(sig_b, 'B_Vol_Crush_After_Spikes')
results['B'] = res_b
print(f"    Sharpe={res_b['sharpe']:.2f}  Return={res_b['total_return']:.1%}  DD={res_b['max_drawdown']:.1%}  Trades={res_b['n_trades']}  {res_b['verdict']}")

# Variant C
print("[C] VIX Term Structure Signal...")
sig_c = variant_c_signals()
res_c = backtest_variant(sig_c, 'C_VIX_Term_Structure')
results['C'] = res_c
print(f"    Sharpe={res_c['sharpe']:.2f}  Return={res_c['total_return']:.1%}  DD={res_c['max_drawdown']:.1%}  Trades={res_c['n_trades']}  {res_c['verdict']}")

# Variant D
print("[D] Tail Hedge Rotation...")
sig_d = variant_d_signals()
res_d = backtest_variant(sig_d, 'D_Tail_Hedge_Rotation')
results['D'] = res_d
print(f"    Sharpe={res_d['sharpe']:.2f}  Return={res_d['total_return']:.1%}  DD={res_d['max_drawdown']:.1%}  Trades={res_d['n_trades']}  {res_d['verdict']}")

# Variant E
print("[E] Vol Risk Premium + Momentum...")
sig_e = variant_e_signals()
res_e = backtest_variant(sig_e, 'E_Vol_Premium_Momentum')
results['E'] = res_e
print(f"    Sharpe={res_e['sharpe']:.2f}  Return={res_e['total_return']:.1%}  DD={res_e['max_drawdown']:.1%}  Trades={res_e['n_trades']}  {res_e['verdict']}")

# Find best variant trade count for adversarial
best_key = max(['A', 'B', 'C', 'D', 'E'], key=lambda k: results[k]['sharpe'])
best_trades = results[best_key]['n_trades']
print(f"\n[F] Adversarial Random (targeting {best_trades} trades to match best={best_key})...")
sig_f = variant_f_signals(best_trades)
res_f = backtest_variant(sig_f, 'F_Adversarial_Random')
results['F'] = res_f
print(f"    Sharpe={res_f['sharpe']:.2f}  Return={res_f['total_return']:.1%}  DD={res_f['max_drawdown']:.1%}  Trades={res_f['n_trades']}  {res_f['verdict']}")

# ─── Summary Table ───
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"{'Var':<4} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>8} {'Trades':>7} {'PF':>6} {'WR':>6} {'CAGR':>7} {'RegGap':>7} {'Perm-p':>7} {'Gates':>6} {'Verdict'}")
print("-"*100)
for k in ['A', 'B', 'C', 'D', 'E', 'F']:
    r = results[k]
    print(f"  {k}   {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['total_return']:>7.1%} {r['max_drawdown']:>7.1%} {r['n_trades']:>7} {r['profit_factor']:>6.2f} {r['win_rate']:>5.0%} {r['cagr']:>6.1%} {r['regime_analysis']['regime_gap']:>7.2f} {r['permutation_p_value']:>7.3f} {r['gates_passed']:>6} {r['verdict']}")

# ─── Save Results ───
output = {
    'metadata': {
        'strategy': 'Volatility Risk Premium Harvesting',
        'start_date': START,
        'end_date': END,
        'starting_capital': CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION,
        'permutation_iterations': PERMUTATION_ITERS,
        'vix3m_source': '63-day MA proxy' if vix3m_approx else 'actual ^VIX3M',
        'run_timestamp': datetime.now().isoformat()
    },
    'variants': results,
    'best_variant': best_key,
    'validation_gates': [
        'Sharpe > 0.5',
        'Permutation test p < 0.05',
        'Regime gap < 0.5',
        'Max drawdown > -50%',
        '>= 20 trades'
    ]
}

output_path = '/home/jupiter/Lvl3Quant/data/vol_risk_premium_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# Print gate details for each variant
print("\n" + "="*70)
print("GATE DETAILS")
print("="*70)
for k in ['A', 'B', 'C', 'D', 'E', 'F']:
    r = results[k]
    print(f"\n  Variant {k} ({r['variant']}):")
    print(f"    Final equity: ${r['final_equity']:.2f}")
    for gate, passed in r['gates'].items():
        status = 'PASS' if passed else 'FAIL'
        print(f"    [{status}] {gate}")
