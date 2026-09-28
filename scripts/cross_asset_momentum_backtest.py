#!/usr/bin/env python3
"""
Cross-Asset Momentum Backtest
==============================
Uses signals from bonds, gold, oil, and the dollar to predict equity direction.
Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
Account: $645 Robinhood, $0 commission, 0.02% slippage on shares.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0       # $0 on shares
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'
DATA_START = '2021-01-01'  # extra lookback for indicators
MOM_WINDOW = 20  # 20-day momentum
SMA_200 = 200
PERM_ITERATIONS = 1000
RISK_FREE_RATE = 0.04  # ~4% for recent years

# ── Data Download ──────────────────────────────────────────────────────────
TICKERS = [
    'SPY', 'QQQ', 'TLT', 'IEF', 'GLD', 'SLV', 'USO', 'DBA', 'UUP',
    'XLE', 'XLF', 'XLK', 'XLV', 'AMZN', 'GOOGL', 'META'
]

print("Downloading data...")
data = yf.download(TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
close = data['Close'].dropna(how='all')
close = close.ffill().dropna()
print(f"Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")

# ── Derived Signals ────────────────────────────────────────────────────────
mom = close.pct_change(MOM_WINDOW)  # 20-day momentum
sma200 = close['SPY'].rolling(SMA_200).mean()

# Regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA
regime = pd.Series(np.where(close['SPY'] > sma200, 'Bull', 'Bear'), index=close.index)

# ── Helper Functions ───────────────────────────────────────────────────────

def apply_slippage(price, direction):
    """Apply slippage: buy higher, sell lower."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    else:
        return price * (1 - SLIPPAGE_PCT)


def run_backtest(signal_series, trade_ticker, close_df, account=ACCOUNT_SIZE,
                 hold_days=None, variant_name=""):
    """
    Run a long-only share backtest given a binary signal.
    signal_series: pd.Series of 1 (long) or 0 (cash), indexed by date.
    trade_ticker: which ticker to trade.
    hold_days: if set, hold for exactly N days after entry regardless of signal.
    Returns dict with equity curve, trades, metrics.
    """
    oot_mask = (close_df.index >= OOT_START) & (close_df.index <= OOT_END)
    dates = close_df.index[oot_mask]
    prices = close_df.loc[dates, trade_ticker]
    signals = signal_series.reindex(dates).fillna(0).astype(int)
    regimes = regime.reindex(dates).fillna('Unknown')

    equity = account
    equity_curve = []
    trades = []
    position = 0  # shares held
    entry_price = 0
    entry_date = None
    hold_until = None

    for i, date in enumerate(dates):
        price = prices.iloc[i]
        sig = signals.iloc[i]

        # Check hold expiry
        if hold_until is not None and date >= hold_until:
            sig = 0  # force exit
            hold_until = None
        elif hold_until is not None:
            sig = 1  # keep holding

        if position == 0 and sig == 1:
            # Enter: buy shares with full equity
            buy_price = apply_slippage(price, 'buy')
            shares = int(equity / buy_price)
            if shares > 0:
                position = shares
                entry_price = buy_price
                entry_date = date
                equity -= shares * buy_price
                if hold_days:
                    hold_until = date + timedelta(days=hold_days)

        elif position > 0 and sig == 0:
            # Exit
            sell_price = apply_slippage(price, 'sell')
            proceeds = position * sell_price
            pnl = proceeds - (position * entry_price)
            pnl_pct = pnl / (position * entry_price) * 100
            trades.append({
                'entry_date': str(entry_date.date()),
                'exit_date': str(date.date()),
                'ticker': trade_ticker,
                'shares': position,
                'entry_price': round(entry_price, 2),
                'exit_price': round(sell_price, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'regime': regimes.loc[entry_date] if entry_date in regimes.index else 'Unknown'
            })
            equity += proceeds
            position = 0
            entry_price = 0
            entry_date = None

        # Mark to market
        mtm = equity + (position * price if position > 0 else 0)
        equity_curve.append({'date': str(date.date()), 'equity': round(mtm, 2), 'regime': regimes.iloc[i]})

    # Close any open position at end
    if position > 0:
        sell_price = apply_slippage(prices.iloc[-1], 'sell')
        proceeds = position * sell_price
        pnl = proceeds - (position * entry_price)
        trades.append({
            'entry_date': str(entry_date.date()),
            'exit_date': str(dates[-1].date()),
            'ticker': trade_ticker,
            'shares': position,
            'entry_price': round(entry_price, 2),
            'exit_price': round(sell_price, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl / (position * entry_price) * 100, 2),
            'regime': regimes.loc[entry_date] if entry_date in regimes.index else 'Unknown'
        })
        equity = equity + proceeds

    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_equity': round(equity if position == 0 else equity + position * prices.iloc[-1], 2),
        'variant': variant_name
    }


def compute_metrics(result, account=ACCOUNT_SIZE):
    """Compute Sharpe, Sortino, PF, WR, MaxDD from backtest result."""
    ec = pd.DataFrame(result['equity_curve'])
    if ec.empty:
        return {'sharpe': 0, 'sortino': 0, 'profit_factor': 0, 'win_rate': 0,
                'max_dd_pct': 0, 'n_trades': 0, 'total_return_pct': 0, 'cagr': 0}

    ec['equity'] = ec['equity'].astype(float)
    ec['date'] = pd.to_datetime(ec['date'])

    # Daily returns
    daily_ret = ec['equity'].pct_change().dropna()

    # Annualized Sharpe
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() - RISK_FREE_RATE/252) / daily_ret.std() * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_ret.mean() - RISK_FREE_RATE/252) / downside.std() * np.sqrt(252)
    else:
        sortino = 0

    # Max drawdown
    peak = ec['equity'].cummax()
    dd = (ec['equity'] - peak) / peak
    max_dd = dd.min() * 100

    # Trade-level stats
    trades = result['trades']
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t['pnl'] > 0]
        losses = [t for t in trades if t['pnl'] <= 0]
        win_rate = len(wins) / n_trades * 100
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.01
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    else:
        win_rate = 0
        profit_factor = 0

    # Total return and CAGR
    total_return = (result['final_equity'] - account) / account * 100
    n_years = max((ec['date'].iloc[-1] - ec['date'].iloc[0]).days / 365.25, 0.01)
    cagr = ((result['final_equity'] / account) ** (1/n_years) - 1) * 100

    # Regime stratification
    bull_trades = [t for t in trades if t.get('regime') == 'Bull']
    bear_trades = [t for t in trades if t.get('regime') == 'Bear']

    bull_wr = len([t for t in bull_trades if t['pnl'] > 0]) / max(len(bull_trades), 1) * 100
    bear_wr = len([t for t in bear_trades if t['pnl'] > 0]) / max(len(bear_trades), 1) * 100

    bull_pnl = sum(t['pnl'] for t in bull_trades)
    bear_pnl = sum(t['pnl'] for t in bear_trades)

    # Per-regime Sharpe (approximate from trade PnLs)
    def trade_sharpe(trade_list):
        if len(trade_list) < 2:
            return 0
        pnls = [t['pnl_pct'] for t in trade_list]
        if np.std(pnls) == 0:
            return 0
        return np.mean(pnls) / np.std(pnls) * np.sqrt(min(len(pnls), 252))

    bull_sharpe = trade_sharpe(bull_trades)
    bear_sharpe = trade_sharpe(bear_trades)

    # Regime gap
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.01)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(profit_factor, 3),
        'win_rate': round(win_rate, 1),
        'max_dd_pct': round(max_dd, 1),
        'n_trades': n_trades,
        'total_return_pct': round(total_return, 1),
        'cagr_pct': round(cagr, 1),
        'final_equity': result['final_equity'],
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'bull_win_rate': round(bull_wr, 1),
        'bear_win_rate': round(bear_wr, 1),
        'bull_pnl': round(bull_pnl, 2),
        'bear_pnl': round(bear_pnl, 2),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'regime_gap': round(regime_gap, 3)
    }


def permutation_test(signal_series, trade_ticker, close_df, actual_sharpe,
                     n_iter=PERM_ITERATIONS, hold_days=None):
    """Shuffle entry dates and compute p-value."""
    oot_mask = (close_df.index >= OOT_START) & (close_df.index <= OOT_END)
    dates = close_df.index[oot_mask]
    sig = signal_series.reindex(dates).fillna(0).astype(int)

    # Get transition points (entries)
    entries = sig.diff().fillna(0)
    entry_dates = entries[entries == 1].index
    n_entries = len(entry_dates)

    if n_entries < 5:
        return 1.0  # not enough trades to test

    perm_sharpes = []
    valid_dates = dates[MOM_WINDOW:]  # exclude warmup

    for _ in range(n_iter):
        # Random entry dates
        rand_entries = np.random.choice(len(valid_dates), size=n_entries, replace=False)
        rand_sig = pd.Series(0, index=dates)

        for idx in rand_entries:
            hold = hold_days if hold_days else 20
            start = valid_dates[idx]
            end_idx = min(idx + hold, len(valid_dates) - 1)
            end = valid_dates[end_idx]
            rand_sig.loc[start:end] = 1

        res = run_backtest(rand_sig, trade_ticker, close_df)
        met = compute_metrics(res)
        perm_sharpes.append(met['sharpe'])

    p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
    return round(p_value, 4)


def validate_5gate(metrics, p_value):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': p_value < 0.05,
        'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
        'max_dd_gt_neg50': metrics['max_dd_pct'] > -50,
        'min_20_trades': metrics['n_trades'] >= 20
    }
    gates['all_pass'] = all(gates.values())
    gates['p_value'] = p_value
    return gates


# ══════════════════════════════════════════════════════════════════════════
# VARIANT A: Bond Signal — TLT momentum → buy QQQ
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ Variant A: Bond Signal (TLT momentum → QQQ) ═══")
sig_a = (mom['TLT'] > 0).astype(int)
res_a = run_backtest(sig_a, 'QQQ', close, variant_name='A_Bond_Signal')
met_a = compute_metrics(res_a)
print(f"  Trades: {met_a['n_trades']}, Sharpe: {met_a['sharpe']}, WR: {met_a['win_rate']}%")
print(f"  Return: {met_a['total_return_pct']}%, MaxDD: {met_a['max_dd_pct']}%")

# ══════════════════════════════════════════════════════════════════════════
# VARIANT B: Gold Risk Signal — GLD up + SPY down → buy SPY (mean reversion)
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ Variant B: Gold Risk Signal (mean-reversion) ═══")
sig_b = ((mom['GLD'] > 0) & (mom['SPY'] < 0)).astype(int)
res_b = run_backtest(sig_b, 'SPY', close, hold_days=10, variant_name='B_Gold_Risk')
met_b = compute_metrics(res_b)
print(f"  Trades: {met_b['n_trades']}, Sharpe: {met_b['sharpe']}, WR: {met_b['win_rate']}%")
print(f"  Return: {met_b['total_return_pct']}%, MaxDD: {met_b['max_dd_pct']}%")

# ══════════════════════════════════════════════════════════════════════════
# VARIANT C: Dollar Weakness — UUP negative momentum → buy growth basket
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ Variant C: Dollar Weakness → Growth Stocks ═══")
# Trade QQQ as growth proxy (individual stocks harder to size with $645)
sig_c = (mom['UUP'] < 0).astype(int)
res_c = run_backtest(sig_c, 'QQQ', close, variant_name='C_Dollar_Weakness')
met_c = compute_metrics(res_c)
print(f"  Trades: {met_c['n_trades']}, Sharpe: {met_c['sharpe']}, WR: {met_c['win_rate']}%")
print(f"  Return: {met_c['total_return_pct']}%, MaxDD: {met_c['max_dd_pct']}%")

# ══════════════════════════════════════════════════════════════════════════
# VARIANT D: Oil-Equity Divergence — USO up >10% but SPY flat/down → buy XLE
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ Variant D: Oil-Equity Divergence → XLE ═══")
uso_surge = mom['USO'] > 0.10
spy_weak = mom['SPY'] <= 0
sig_d = (uso_surge & spy_weak).astype(int)
res_d = run_backtest(sig_d, 'XLE', close, hold_days=20, variant_name='D_Oil_Divergence')
met_d = compute_metrics(res_d)
print(f"  Trades: {met_d['n_trades']}, Sharpe: {met_d['sharpe']}, WR: {met_d['win_rate']}%")
print(f"  Return: {met_d['total_return_pct']}%, MaxDD: {met_d['max_dd_pct']}%")

# ══════════════════════════════════════════════════════════════════════════
# VARIANT E: Multi-Asset Momentum Score — 3/4 asset classes positive → QQQ
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ Variant E: Multi-Asset Momentum (3/4 risk-on → QQQ) ═══")
score = pd.DataFrame({
    'bonds': (mom['TLT'] > 0).astype(int),
    'gold': (mom['GLD'] > 0).astype(int),
    'equity': (mom['SPY'] > 0).astype(int),
    'dollar_inv': (mom['UUP'] < 0).astype(int),  # weak dollar = risk-on
}).sum(axis=1)
sig_e_long = (score >= 3).astype(int)
# When 3/4 negative → buy TLT (risk-off)
sig_e_safe = (score <= 1).astype(int)

# Run risk-on leg
res_e_on = run_backtest(sig_e_long, 'QQQ', close, variant_name='E_MultiAsset_RiskOn')
met_e_on = compute_metrics(res_e_on)
# Run risk-off leg
res_e_off = run_backtest(sig_e_safe, 'TLT', close, variant_name='E_MultiAsset_RiskOff')
met_e_off = compute_metrics(res_e_off)

# Combined: merge equity curves
ec_on = pd.DataFrame(res_e_on['equity_curve'])
ec_off = pd.DataFrame(res_e_off['equity_curve'])
if not ec_on.empty and not ec_off.empty:
    ec_on['date'] = pd.to_datetime(ec_on['date'])
    ec_off['date'] = pd.to_datetime(ec_off['date'])
    ec_on = ec_on.set_index('date')
    ec_off = ec_off.set_index('date')
    # Each starts at 645, so combined is 645 base with returns from both
    ret_on = ec_on['equity'].astype(float).pct_change().fillna(0)
    ret_off = ec_off['equity'].astype(float).pct_change().fillna(0)
    # They alternate (non-overlapping signals), so combine
    combined_ret = ret_on + ret_off  # only one active at a time
    combined_eq = ACCOUNT_SIZE * (1 + combined_ret).cumprod()
    combined_sharpe = ((combined_ret.mean() - RISK_FREE_RATE/252) / combined_ret.std() * np.sqrt(252)) if combined_ret.std() > 0 else 0
else:
    combined_sharpe = 0

print(f"  Risk-On leg: Trades={met_e_on['n_trades']}, Sharpe={met_e_on['sharpe']}")
print(f"  Risk-Off leg: Trades={met_e_off['n_trades']}, Sharpe={met_e_off['sharpe']}")
print(f"  Combined Sharpe: {round(combined_sharpe, 3)}")

# For validation, use risk-on leg as primary
met_e = met_e_on
met_e['combined_sharpe'] = round(combined_sharpe, 3)
res_e = res_e_on

# ══════════════════════════════════════════════════════════════════════════
# VARIANT F: Yield Curve Proxy — TLT outperforms IEF → buy XLF
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ Variant F: Yield Curve Proxy (TLT vs IEF → XLF) ═══")
tlt_vs_ief = (close['TLT'] / close['IEF']).pct_change(MOM_WINDOW)
sig_f = (tlt_vs_ief > 0).astype(int)  # TLT outperforming = curve steepening
res_f = run_backtest(sig_f, 'XLF', close, variant_name='F_Yield_Curve')
met_f = compute_metrics(res_f)
print(f"  Trades: {met_f['n_trades']}, Sharpe: {met_f['sharpe']}, WR: {met_f['win_rate']}%")
print(f"  Return: {met_f['total_return_pct']}%, MaxDD: {met_f['max_dd_pct']}%")

# ══════════════════════════════════════════════════════════════════════════
# PERMUTATION TESTS (1000 iterations each)
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ Running Permutation Tests (1000 iter each) ═══")
variants = [
    ('A', sig_a, 'QQQ', met_a, None),
    ('B', sig_b, 'SPY', met_b, 10),
    ('C', sig_c, 'QQQ', met_c, None),
    ('D', sig_d, 'XLE', met_d, 20),
    ('E', sig_e_long, 'QQQ', met_e, None),
    ('F', sig_f, 'XLF', met_f, None),
]

p_values = {}
for name, sig, ticker, met, hd in variants:
    print(f"  Variant {name}...", end=' ', flush=True)
    p = permutation_test(sig, ticker, close, met['sharpe'], hold_days=hd)
    p_values[name] = p
    print(f"p={p}")

# ══════════════════════════════════════════════════════════════════════════
# 5-GATE VALIDATION
# ══════════════════════════════════════════════════════════════════════════
print("\n═══ 5-Gate Validation ═══")
all_results = {}
variant_data = [
    ('A_Bond_Signal', met_a, res_a, p_values['A']),
    ('B_Gold_Risk', met_b, res_b, p_values['B']),
    ('C_Dollar_Weakness', met_c, res_c, p_values['C']),
    ('D_Oil_Divergence', met_d, res_d, p_values['D']),
    ('E_MultiAsset', met_e, res_e, p_values['E']),
    ('F_Yield_Curve', met_f, res_f, p_values['F']),
]

for name, met, res, pv in variant_data:
    gates = validate_5gate(met, pv)
    passed = sum(1 for k, v in gates.items() if k != 'all_pass' and k != 'p_value' and v)

    print(f"\n  {name}: {'PASS' if gates['all_pass'] else 'FAIL'} ({passed}/5 gates)")
    print(f"    Sharpe={met['sharpe']} {'OK' if gates['sharpe_gt_0.5'] else 'FAIL'}")
    print(f"    Perm p={pv} {'OK' if gates['perm_p_lt_0.05'] else 'FAIL'}")
    print(f"    Regime gap={met['regime_gap']} {'OK' if gates['regime_gap_lt_0.5'] else 'FAIL'}")
    print(f"    MaxDD={met['max_dd_pct']}% {'OK' if gates['max_dd_gt_neg50'] else 'FAIL'}")
    print(f"    Trades={met['n_trades']} {'OK' if gates['min_20_trades'] else 'FAIL'}")
    print(f"    Bull WR={met['bull_win_rate']}% ({met['bull_trades']} trades, PnL=${met['bull_pnl']})")
    print(f"    Bear WR={met['bear_win_rate']}% ({met['bear_trades']} trades, PnL=${met['bear_pnl']})")

    all_results[name] = {
        'metrics': met,
        'gates': gates,
        'trades': res['trades'],
        'equity_curve_start': res['equity_curve'][:3] if res['equity_curve'] else [],
        'equity_curve_end': res['equity_curve'][-3:] if res['equity_curve'] else [],
        'n_equity_points': len(res['equity_curve'])
    }

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY & RANKING
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("CROSS-ASSET MOMENTUM — FINAL RANKING")
print("="*70)
print(f"{'Variant':<25} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR%':>5} {'DD%':>6} {'Ret%':>7} {'#Tr':>4} {'Pass':>5}")
print("-"*70)

ranked = sorted(variant_data, key=lambda x: x[1]['sharpe'], reverse=True)
for name, met, res, pv in ranked:
    gates = validate_5gate(met, pv)
    status = 'PASS' if gates['all_pass'] else 'FAIL'
    print(f"{name:<25} {met['sharpe']:>7.3f} {met['sortino']:>8.3f} {met['profit_factor']:>6.2f} "
          f"{met['win_rate']:>5.1f} {met['max_dd_pct']:>6.1f} {met['total_return_pct']:>7.1f} "
          f"{met['n_trades']:>4d} {status:>5}")

# Best variant
best_name, best_met, best_res, best_pv = ranked[0]
print(f"\nBest variant: {best_name} (Sharpe={best_met['sharpe']}, CAGR={best_met['cagr_pct']}%)")
passers = [name for name, met, res, pv in variant_data if validate_5gate(met, pv)['all_pass']]
print(f"Passed 5-gate: {passers if passers else 'NONE'}")

# ══════════════════════════════════════════════════════════════════════════
# SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════
output = {
    'metadata': {
        'strategy': 'Cross-Asset Momentum',
        'account_size': ACCOUNT_SIZE,
        'oot_period': f'{OOT_START} to {OOT_END}',
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION,
        'momentum_window': MOM_WINDOW,
        'permutation_iterations': PERM_ITERATIONS,
        'run_date': str(datetime.now()),
        'data_range': f"{close.index[0].date()} to {close.index[-1].date()}"
    },
    'variants': all_results,
    'ranking': [
        {
            'rank': i+1,
            'variant': name,
            'sharpe': met['sharpe'],
            'sortino': met['sortino'],
            'profit_factor': met['profit_factor'],
            'win_rate': met['win_rate'],
            'max_dd_pct': met['max_dd_pct'],
            'total_return_pct': met['total_return_pct'],
            'cagr_pct': met['cagr_pct'],
            'n_trades': met['n_trades'],
            'p_value': pv,
            'passed_5gate': validate_5gate(met, pv)['all_pass']
        }
        for i, (name, met, res, pv) in enumerate(ranked)
    ],
    'passed_variants': passers,
    'best_variant': best_name
}

out_path = Path('/home/jupiter/Lvl3Quant/data/cross_asset_momentum_results.json')
out_path.parent.mkdir(parents=True, exist_ok=True)
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)
print(f"\nResults saved to {out_path}")
print("Done.")
