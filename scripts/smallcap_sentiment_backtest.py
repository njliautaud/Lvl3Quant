#!/usr/bin/env python3
"""
Small-Cap Value + Sentiment Signal Backtest
Searching for strategies UNCORRELATED to QQQ.
6 variants: A-F, with 5-gate validation framework.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = '2022-01-01'
OOT_END = '2026-07-29'

TICKERS = ['IWN', 'IWM', 'IWD', 'VTV', 'SLYV', 'SPY', 'QQQ', '^VIX', 'TLT']

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
data = yf.download(TICKERS, start='2021-01-01', end=OOT_END, auto_adjust=True, progress=False)

close = data['Close'].copy()
close.columns = [c if isinstance(c, str) else c[0] for c in close.columns]
# Rename ^VIX
if '^VIX' in close.columns:
    close.rename(columns={'^VIX': 'VIX'}, inplace=True)

close = close.ffill().dropna()
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} rows")

# Trim to OOT period for trading (keep earlier data for lookback)
oot_mask = close.index >= OOT_START

# ── Helper Functions ────────────────────────────────────────────────────────
def apply_slippage(price, direction='buy'):
    """Apply slippage to execution price."""
    if direction == 'buy':
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def backtest_signals(close_df, signals, asset_col, label):
    """
    Generic backtester.
    signals: Series of 1 (long asset_col), 0 (cash), or ticker string for rotation.
    Returns dict with metrics.
    """
    oot_idx = close_df.index[close_df.index >= OOT_START]

    equity = CAPITAL
    equity_curve = []
    trades = []
    position = None  # None or (ticker, entry_price, entry_date)
    hold_counter = 0

    for i, dt in enumerate(oot_idx):
        sig = signals.get(dt, 0)

        # Determine target asset
        if isinstance(sig, str):
            target = sig
        elif sig == 1:
            target = asset_col
        else:
            target = None

        # Close position if signal says cash or different asset
        if position is not None:
            cur_ticker = position[0]
            if target != cur_ticker:
                exit_price = apply_slippage(close_df.loc[dt, cur_ticker], 'sell')
                pnl_pct = (exit_price / position[1]) - 1
                equity *= (1 + pnl_pct)
                trades.append({
                    'entry': str(position[2].date()),
                    'exit': str(dt.date()),
                    'ticker': cur_ticker,
                    'pnl_pct': pnl_pct
                })
                position = None

        # Open new position
        if target is not None and position is None:
            if target in close_df.columns:
                entry_price = apply_slippage(close_df.loc[dt, target], 'buy')
                position = (target, entry_price, dt)

        # Mark to market
        if position is not None:
            cur_price = close_df.loc[dt, position[0]]
            mtm = equity * (cur_price / position[1])
            equity_curve.append(mtm)
        else:
            equity_curve.append(equity)

    # Close any remaining position
    if position is not None:
        last_dt = oot_idx[-1]
        exit_price = apply_slippage(close_df.loc[last_dt, position[0]], 'sell')
        pnl_pct = (exit_price / position[1]) - 1
        equity *= (1 + pnl_pct)
        trades.append({
            'entry': str(position[2].date()),
            'exit': str(last_dt.date()),
            'ticker': position[0],
            'pnl_pct': pnl_pct
        })

    eq = pd.Series(equity_curve, index=oot_idx[:len(equity_curve)])
    return compute_metrics(eq, trades, close_df, label)


def compute_metrics(eq, trades, close_df, label):
    """Compute performance metrics and validation gates."""
    oot_idx = close_df.index[close_df.index >= OOT_START]

    # Daily returns
    daily_ret = eq.pct_change().dropna()

    # Annualized metrics
    n_years = len(daily_ret) / 252
    total_ret = (eq.iloc[-1] / eq.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252) if len(daily_ret) > 1 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else 0
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    # Trade stats
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        losses = [t for t in trades if t['pnl_pct'] <= 0]
        avg_loss = abs(np.mean([t['pnl_pct'] for t in losses])) if losses else 0
        profit_factor = (sum(t['pnl_pct'] for t in wins) / abs(sum(t['pnl_pct'] for t in losses))) if losses and sum(t['pnl_pct'] for t in losses) != 0 else float('inf')
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    # QQQ correlation
    qqq_ret = close_df['QQQ'].loc[oot_idx].pct_change().dropna()
    common_idx = daily_ret.index.intersection(qqq_ret.index)
    if len(common_idx) > 10:
        qqq_corr = daily_ret.loc[common_idx].corr(qqq_ret.loc[common_idx])
    else:
        qqq_corr = 0

    # SPY correlation
    spy_ret = close_df['SPY'].loc[oot_idx].pct_change().dropna()
    common_idx_spy = daily_ret.index.intersection(spy_ret.index)
    if len(common_idx_spy) > 10:
        spy_corr = daily_ret.loc[common_idx_spy].corr(spy_ret.loc[common_idx_spy])
    else:
        spy_corr = 0

    # Permutation test (1000 shuffles)
    if len(daily_ret) > 20:
        obs_sharpe = sharpe
        count_better = 0
        shuffled_rets = daily_ret.values.copy()
        for _ in range(1000):
            np.random.shuffle(shuffled_rets)
            s_ret = np.mean(shuffled_rets) * 252
            s_vol = np.std(shuffled_rets) * np.sqrt(252)
            s_sharpe = s_ret / s_vol if s_vol > 0 else 0
            if s_sharpe >= obs_sharpe:
                count_better += 1
        perm_p = count_better / 1000
    else:
        perm_p = 1.0

    # Regime analysis: classify days by SPY return
    spy_20d = close_df['SPY'].pct_change(20)
    regime_green = spy_20d > 0.02
    regime_red = spy_20d < -0.02
    regime_flat = ~regime_green & ~regime_red

    green_days = daily_ret.index.intersection(regime_green[regime_green].index)
    red_days = daily_ret.index.intersection(regime_red[regime_red].index)

    if len(green_days) > 5:
        green_sharpe = daily_ret.loc[green_days].mean() * 252 / (daily_ret.loc[green_days].std() * np.sqrt(252)) if daily_ret.loc[green_days].std() > 0 else 0
    else:
        green_sharpe = 0
    if len(red_days) > 5:
        red_sharpe = daily_ret.loc[red_days].mean() * 252 / (daily_ret.loc[red_days].std() * np.sqrt(252)) if daily_ret.loc[red_days].std() > 0 else 0
    else:
        red_sharpe = 0

    max_sharpe = max(abs(green_sharpe), abs(red_sharpe))
    regime_gap = abs(green_sharpe - red_sharpe) / max_sharpe if max_sharpe > 0 else 0

    # 5-gate validation
    gate_sharpe = sharpe > 0.5
    gate_perm = perm_p < 0.05
    gate_regime = regime_gap < 0.5
    gate_mdd = max_dd > -0.50
    gate_trades = n_trades >= 20
    gates_passed = sum([gate_sharpe, gate_perm, gate_regime, gate_mdd, gate_trades])
    all_gates = gates_passed == 5

    return {
        'label': label,
        'total_return_pct': round(total_ret * 100, 2),
        'ann_return_pct': round(ann_ret * 100, 2),
        'ann_vol_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'profit_factor': round(profit_factor, 3) if profit_factor != float('inf') else 'inf',
        'win_rate': round(win_rate * 100, 1),
        'n_trades': n_trades,
        'avg_win_pct': round(avg_win * 100, 3),
        'avg_loss_pct': round(avg_loss * 100, 3),
        'qqq_correlation': round(qqq_corr, 3),
        'spy_correlation': round(spy_corr, 3),
        'final_equity': round(eq.iloc[-1], 2),
        'green_regime_sharpe': round(green_sharpe, 3),
        'red_regime_sharpe': round(red_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'perm_p_value': round(perm_p, 4),
        'validation_gates': {
            'sharpe_gt_0.5': gate_sharpe,
            'perm_p_lt_0.05': gate_perm,
            'regime_gap_lt_0.5': gate_regime,
            'mdd_gt_neg50pct': gate_mdd,
            'trades_gte_20': gate_trades,
            'gates_passed': gates_passed,
            'all_passed': all_gates
        }
    }


# ── Strategy A: Small-Cap Value Momentum ────────────────────────────────────
print("\n[A] Small-Cap Value Momentum...")
signals_a = {}
for dt in close.index[close.index >= OOT_START]:
    loc = close.index.get_loc(dt)
    if loc < 20:
        signals_a[dt] = 0
        continue
    iwn_20d_ret = close['IWN'].iloc[loc] / close['IWN'].iloc[loc - 20] - 1
    vix_val = close['VIX'].iloc[loc]
    if iwn_20d_ret > 0 and vix_val < 25:
        signals_a[dt] = 1
    else:
        signals_a[dt] = 0

# Apply 15-day hold period
hold_a = {}
in_trade = False
hold_count = 0
for dt in close.index[close.index >= OOT_START]:
    sig = signals_a.get(dt, 0)
    if in_trade:
        hold_count += 1
        if hold_count >= 15:
            in_trade = False
            hold_a[dt] = 0
        else:
            hold_a[dt] = 1
    else:
        if sig == 1:
            in_trade = True
            hold_count = 0
            hold_a[dt] = 1
        else:
            hold_a[dt] = 0

result_a = backtest_signals(close, hold_a, 'IWN', 'A: Small-Cap Value Momentum')
print(f"  Sharpe={result_a['sharpe']}, QQQ_corr={result_a['qqq_correlation']}, Trades={result_a['n_trades']}")


# ── Strategy B: Value-Growth Rotation ───────────────────────────────────────
print("[B] Value-Growth Rotation...")
ratio_bg = close['IWN'] / close['QQQ']
ratio_60d_ma = ratio_bg.rolling(60).mean()

signals_b = {}
rebal_count = 0
current_asset = None
for dt in close.index[close.index >= OOT_START]:
    loc = close.index.get_loc(dt)
    if loc < 60:
        signals_b[dt] = 'QQQ'
        continue
    rebal_count += 1
    if rebal_count >= 15 or current_asset is None:
        rebal_count = 0
        if ratio_bg.iloc[loc] > ratio_60d_ma.iloc[loc]:
            current_asset = 'IWN'
        else:
            current_asset = 'QQQ'
    signals_b[dt] = current_asset

result_b = backtest_signals(close, signals_b, 'IWN', 'B: Value-Growth Rotation')
print(f"  Sharpe={result_b['sharpe']}, QQQ_corr={result_b['qqq_correlation']}, Trades={result_b['n_trades']}")


# ── Strategy C: Small vs Large Cap Rotation ─────────────────────────────────
print("[C] Small vs Large Cap Rotation...")
signals_c = {}
rebal_count = 0
current_asset = None
for dt in close.index[close.index >= OOT_START]:
    loc = close.index.get_loc(dt)
    if loc < 20:
        signals_c[dt] = 'SPY'
        continue
    rebal_count += 1
    if rebal_count >= 10 or current_asset is None:
        rebal_count = 0
        iwm_20d = close['IWM'].iloc[loc] / close['IWM'].iloc[loc - 20] - 1
        spy_20d = close['SPY'].iloc[loc] / close['SPY'].iloc[loc - 20] - 1
        if iwm_20d > spy_20d:
            current_asset = 'IWM'
        else:
            current_asset = 'SPY'
    signals_c[dt] = current_asset

result_c = backtest_signals(close, signals_c, 'IWM', 'C: Small vs Large Cap Rotation')
print(f"  Sharpe={result_c['sharpe']}, QQQ_corr={result_c['qqq_correlation']}, Trades={result_c['n_trades']}")


# ── Strategy D: Deep Value Contrarian ───────────────────────────────────────
print("[D] Deep Value Contrarian...")
signals_d = {}
in_trade = False
hold_count = 0
for dt in close.index[close.index >= OOT_START]:
    loc = close.index.get_loc(dt)
    if loc < 20:
        signals_d[dt] = 0
        continue
    if in_trade:
        hold_count += 1
        if hold_count >= 20:
            in_trade = False
            signals_d[dt] = 0
        else:
            signals_d[dt] = 1
    else:
        iwn_20d = close['IWN'].iloc[loc] / close['IWN'].iloc[loc - 20] - 1
        vix_val = close['VIX'].iloc[loc]
        if iwn_20d < -0.05 and vix_val > 22:
            in_trade = True
            hold_count = 0
            signals_d[dt] = 1
        else:
            signals_d[dt] = 0

result_d = backtest_signals(close, signals_d, 'IWN', 'D: Deep Value Contrarian')
print(f"  Sharpe={result_d['sharpe']}, QQQ_corr={result_d['qqq_correlation']}, Trades={result_d['n_trades']}")


# ── Strategy E: Value + Rate Signal ─────────────────────────────────────────
print("[E] Value + Rate Signal...")
tlt_20d_ret = close['TLT'].pct_change(20)
iwn_20d_ret = close['IWN'].pct_change(20)

signals_e = {}
in_trade = False
hold_count = 0
current_asset = None
for dt in close.index[close.index >= OOT_START]:
    loc = close.index.get_loc(dt)
    if loc < 20:
        signals_e[dt] = 0
        continue
    if in_trade:
        hold_count += 1
        if hold_count >= 15:
            in_trade = False
            signals_e[dt] = 0
            current_asset = None
        else:
            signals_e[dt] = current_asset
    else:
        tlt_up = tlt_20d_ret.iloc[loc] > 0
        iwn_mom_pos = iwn_20d_ret.iloc[loc] > 0
        if tlt_up and iwn_mom_pos:
            in_trade = True
            hold_count = 0
            current_asset = 'IWN'
            signals_e[dt] = 'IWN'
        elif not tlt_up:
            in_trade = True
            hold_count = 0
            current_asset = 'VTV'
            signals_e[dt] = 'VTV'
        else:
            signals_e[dt] = 0

result_e = backtest_signals(close, signals_e, 'IWN', 'E: Value + Rate Signal')
print(f"  Sharpe={result_e['sharpe']}, QQQ_corr={result_e['qqq_correlation']}, Trades={result_e['n_trades']}")


# ── Strategy F: Multi-Factor Value Score ────────────────────────────────────
print("[F] Multi-Factor Value Score...")
signals_f = {}
rebal_count = 0
prev_signal = 0
for dt in close.index[close.index >= OOT_START]:
    loc = close.index.get_loc(dt)
    if loc < 60:
        signals_f[dt] = 0
        continue
    rebal_count += 1
    if rebal_count >= 10 or prev_signal == 0:
        rebal_count = 0
        # Score components
        score = 0
        # 1. IWN 1m momentum > 0
        iwn_1m = close['IWN'].iloc[loc] / close['IWN'].iloc[loc - 20] - 1
        if iwn_1m > 0:
            score += 1
        # 2. IWN/QQQ ratio vs 60d MA (value outperforming)
        ratio_val = ratio_bg.iloc[loc]
        ratio_ma = ratio_60d_ma.iloc[loc]
        if pd.notna(ratio_ma) and ratio_val > ratio_ma:
            score += 1
        # 3. VIX < 22
        if close['VIX'].iloc[loc] < 22:
            score += 1
        # 4. TLT uptrend (20d)
        tlt_20 = close['TLT'].iloc[loc] / close['TLT'].iloc[loc - 20] - 1
        if tlt_20 > 0:
            score += 1

        if score >= 2:
            prev_signal = 1
            signals_f[dt] = 1
        elif score <= 0:
            prev_signal = 0
            signals_f[dt] = 0
        else:
            signals_f[dt] = prev_signal
    else:
        signals_f[dt] = prev_signal

result_f = backtest_signals(close, signals_f, 'IWN', 'F: Multi-Factor Value Score')
print(f"  Sharpe={result_f['sharpe']}, QQQ_corr={result_f['qqq_correlation']}, Trades={result_f['n_trades']}")


# ── Benchmark: Buy & Hold ──────────────────────────────────────────────────
print("\n[Benchmarks]...")
oot_close = close[close.index >= OOT_START]

benchmarks = {}
for ticker in ['IWN', 'QQQ', 'SPY', 'IWM']:
    bh_ret = oot_close[ticker].iloc[-1] / oot_close[ticker].iloc[0] - 1
    bh_daily = oot_close[ticker].pct_change().dropna()
    bh_sharpe = (bh_daily.mean() * 252) / (bh_daily.std() * np.sqrt(252)) if bh_daily.std() > 0 else 0
    bh_peak = oot_close[ticker].expanding().max()
    bh_dd = ((oot_close[ticker] - bh_peak) / bh_peak).min()
    benchmarks[ticker] = {
        'total_return_pct': round(bh_ret * 100, 2),
        'sharpe': round(bh_sharpe, 3),
        'max_drawdown_pct': round(bh_dd * 100, 2)
    }
    print(f"  {ticker} B&H: Return={bh_ret*100:.1f}%, Sharpe={bh_sharpe:.3f}, MDD={bh_dd*100:.1f}%")


# ── Compile Results ─────────────────────────────────────────────────────────
all_results = {
    'metadata': {
        'description': 'Small-Cap Value + Sentiment Signal Backtest',
        'purpose': 'Find strategies UNCORRELATED to QQQ (Sharpe 2.38)',
        'oot_period': f'{OOT_START} to {OOT_END}',
        'capital': CAPITAL,
        'slippage_pct': SLIPPAGE_PCT,
        'commission': COMMISSION,
        'run_timestamp': datetime.now().isoformat()
    },
    'strategies': {
        'A': result_a,
        'B': result_b,
        'C': result_c,
        'D': result_d,
        'E': result_e,
        'F': result_f
    },
    'benchmarks': benchmarks,
    'summary': {
        'best_sharpe': None,
        'lowest_qqq_correlation': None,
        'gates_passed_summary': {},
        'recommendation': None
    }
}

# Find best
strats = all_results['strategies']
best_sharpe_key = max(strats, key=lambda k: strats[k]['sharpe'])
lowest_corr_key = min(strats, key=lambda k: abs(strats[k]['qqq_correlation']))

all_results['summary']['best_sharpe'] = {
    'variant': best_sharpe_key,
    'sharpe': strats[best_sharpe_key]['sharpe'],
    'qqq_corr': strats[best_sharpe_key]['qqq_correlation']
}
all_results['summary']['lowest_qqq_correlation'] = {
    'variant': lowest_corr_key,
    'qqq_corr': strats[lowest_corr_key]['qqq_correlation'],
    'sharpe': strats[lowest_corr_key]['sharpe']
}

for k, v in strats.items():
    all_results['summary']['gates_passed_summary'][k] = {
        'gates_passed': v['validation_gates']['gates_passed'],
        'all_passed': v['validation_gates']['all_passed']
    }

# Recommendation
passing = [k for k, v in strats.items() if v['validation_gates']['all_passed']]
if passing:
    # Among passing, pick lowest QQQ correlation
    best = min(passing, key=lambda k: abs(strats[k]['qqq_correlation']))
    all_results['summary']['recommendation'] = f"Variant {best} passes all 5 gates with QQQ correlation {strats[best]['qqq_correlation']:.3f}"
else:
    # Find one with most gates passed
    most_gates = max(strats, key=lambda k: strats[k]['validation_gates']['gates_passed'])
    all_results['summary']['recommendation'] = f"No variant passes all 5 gates. Best: Variant {most_gates} with {strats[most_gates]['validation_gates']['gates_passed']}/5 gates."

# Save results
output_path = '/home/jupiter/Lvl3Quant/data/smallcap_sentiment_results.json'
with open(output_path, 'w') as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# ── Print Summary Table ────────────────────────────────────────────────────
print("\n" + "=" * 100)
print(f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'Return%':>8} {'MDD%':>7} {'WR%':>6} {'Trades':>7} {'QQQ_r':>7} {'Gates':>6}")
print("=" * 100)
for k in ['A', 'B', 'C', 'D', 'E', 'F']:
    v = strats[k]
    print(f"{v['label']:<35} {v['sharpe']:>7.3f} {v['sortino']:>8.3f} {v['total_return_pct']:>7.1f}% {v['max_drawdown_pct']:>6.1f}% {v['win_rate']:>5.1f} {v['n_trades']:>7} {v['qqq_correlation']:>7.3f} {v['validation_gates']['gates_passed']:>3}/5")
print("=" * 100)
print(f"\nRecommendation: {all_results['summary']['recommendation']}")
