#!/usr/bin/env python3
"""
Round 2 Strategy 3: Risk-On/Risk-Off with 3x Leverage
Composite signal: VIX term structure + market breadth + momentum.
Risk-ON: hold TQQQ/UPRO. Risk-OFF: hold SHY or cash.

Walk-forward sliding window backtest.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import warnings
import os
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/aggressive/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ROUND 2 — STRATEGY 3: RISK-ON/RISK-OFF WITH 3X LEVERAGE")
print("=" * 70)

# Download data
print("\n[1/5] Downloading market data...")
close_dict = {}
for t in ['SPY', 'QQQ', 'TQQQ', 'UPRO', 'SHY', 'TLT', 'GLD']:
    try:
        d = yf.download(t, start='2011-01-01', end='2026-07-01', progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        close_dict[t] = d['Close'].dropna()
        print(f"  {t}: {len(close_dict[t])} days")
    except Exception as e:
        print(f"  {t}: FAILED - {e}")

# VIX separately
try:
    vix_data = yf.download('^VIX', start='2011-01-01', end='2026-07-01', progress=False)
    if isinstance(vix_data.columns, pd.MultiIndex):
        vix_data.columns = vix_data.columns.get_level_values(0)
    close_dict['VIX'] = vix_data['Close'].dropna()
    print(f"  VIX: {len(close_dict['VIX'])} days")
except:
    print("  VIX: FAILED")

close = pd.DataFrame(close_dict).ffill().dropna(how='all')
print(f"  Got data for: {list(close.columns)}")
print(f"  Date range: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")

# SPY for regime classification
spy = close['SPY'] if 'SPY' in close.columns else None
spy_monthly = spy.resample('ME').last() if spy is not None else None
spy_monthly_ret = spy_monthly.pct_change() if spy_monthly is not None else None

regime_map = {}
if spy_monthly_ret is not None:
    for dt, ret in spy_monthly_ret.items():
        if pd.notna(ret):
            regime_map[dt.strftime('%Y-%m')] = 'green' if ret >= 0 else 'red'


def build_signals(close_df):
    """Build risk-on/risk-off signals."""
    signals = pd.DataFrame(index=close_df.index)

    spy = close_df['SPY'] if 'SPY' in close_df.columns else None
    qqq = close_df['QQQ'] if 'QQQ' in close_df.columns else None
    vix = close_df['VIX'] if 'VIX' in close_df.columns else None

    # Signal 1: SPY above/below 200-day MA
    if spy is not None:
        signals['spy_200ma'] = (spy > spy.rolling(200).mean()).astype(float)
        signals['spy_50ma'] = (spy > spy.rolling(50).mean()).astype(float)

    # Signal 2: SPY above/below 50-day MA
    if qqq is not None:
        signals['qqq_200ma'] = (qqq > qqq.rolling(200).mean()).astype(float)

    # Signal 3: VIX level (below 20 = risk-on, above 25 = risk-off)
    if vix is not None:
        signals['vix_low'] = (vix < 20).astype(float)
        signals['vix_high'] = (vix > 25).astype(float)
        signals['vix_signal'] = signals['vix_low'] - signals['vix_high']
        # VIX trend: falling VIX = risk-on
        vix_ma10 = vix.rolling(10).mean()
        vix_ma30 = vix.rolling(30).mean()
        signals['vix_trend'] = (vix_ma10 < vix_ma30).astype(float)

    # Signal 4: Momentum (SPY 1-month return)
    if spy is not None:
        spy_ret_21d = spy.pct_change(21)
        signals['spy_mom_1m'] = (spy_ret_21d > 0).astype(float)
        spy_ret_63d = spy.pct_change(63)
        signals['spy_mom_3m'] = (spy_ret_63d > 0).astype(float)

    # Signal 5: Breadth proxy - ratio of QQQ/SPY (tech leadership)
    if spy is not None and qqq is not None:
        ratio = qqq / spy
        ratio_ma = ratio.rolling(50).mean()
        signals['tech_leadership'] = (ratio > ratio_ma).astype(float)

    # Signal 6: Dual momentum (SPY vs SHY trailing 3m return)
    if spy is not None and 'SHY' in close_df.columns:
        shy = close_df['SHY']
        spy_3m = spy.pct_change(63)
        shy_3m = shy.pct_change(63)
        signals['dual_mom'] = ((spy_3m > shy_3m) & (spy_3m > 0)).astype(float)

    return signals


def risk_on_off_backtest(close_df, signal_cols, threshold=0.5,
                          risk_on_ticker='TQQQ', risk_off_ticker='SHY',
                          initial_capital=441, train_window=252):
    """
    Walk-forward risk-on/risk-off strategy.
    - signal_cols: list of signal columns to combine
    - threshold: fraction of signals needed for risk-on
    - train_window: days to use for calibrating signal weights
    """
    signals = build_signals(close_df)

    available_signals = [s for s in signal_cols if s in signals.columns]
    if len(available_signals) == 0:
        return None

    if risk_on_ticker not in close_df.columns or risk_off_ticker not in close_df.columns:
        return None

    risk_on = close_df[risk_on_ticker]
    risk_off = close_df[risk_off_ticker]

    risk_on_ret = risk_on.pct_change()
    risk_off_ret = risk_off.pct_change()

    # Composite signal
    composite = signals[available_signals].mean(axis=1)

    # Walk-forward: use training window to calibrate threshold
    start_idx = train_window + 200  # Need 200 days for signals to warm up

    daily_returns = pd.Series(0.0, index=close_df.index[start_idx:])
    positions = pd.Series('CASH', index=close_df.index[start_idx:])

    for i in range(start_idx, len(close_df.index)):
        dt = close_df.index[i]

        # Walk-forward: use trailing train_window to find optimal threshold
        # (In practice, just use the composite signal)
        train_start = max(0, i - train_window)
        train_composite = composite.iloc[train_start:i]
        train_risk_on_ret = risk_on_ret.iloc[train_start:i]
        train_risk_off_ret = risk_off_ret.iloc[train_start:i]

        if len(train_composite.dropna()) < 50:
            continue

        # Find threshold that maximized Sharpe in training
        best_thresh = threshold
        best_sharpe = -999
        for test_thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            mask_on = train_composite > test_thresh
            test_rets = pd.Series(0.0, index=train_composite.index)
            test_rets[mask_on] = train_risk_on_ret[mask_on]
            test_rets[~mask_on] = train_risk_off_ret[~mask_on]
            test_rets = test_rets.dropna()
            if len(test_rets) > 20 and test_rets.std() > 0:
                s = test_rets.mean() / test_rets.std() * np.sqrt(252)
                if s > best_sharpe:
                    best_sharpe = s
                    best_thresh = test_thresh

        # Apply to today
        sig_today = composite.iloc[i] if i < len(composite) else np.nan

        if pd.notna(sig_today) and sig_today > best_thresh:
            if dt in risk_on_ret.index and pd.notna(risk_on_ret.loc[dt]):
                daily_returns[dt] = risk_on_ret.loc[dt]
                positions[dt] = risk_on_ticker
        else:
            if dt in risk_off_ret.index and pd.notna(risk_off_ret.loc[dt]):
                daily_returns[dt] = risk_off_ret.loc[dt]
                positions[dt] = risk_off_ticker

    return daily_returns, positions


def compute_metrics_daily(daily_returns, initial_capital=441):
    """Compute metrics from daily returns."""
    rets = daily_returns.dropna().values
    if len(rets) == 0 or np.std(rets) == 0:
        return None

    equity = initial_capital * np.cumprod(1 + rets)
    n_years = len(rets) / 252

    cagr = (equity[-1] / initial_capital) ** (1/n_years) - 1 if n_years > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    neg = rets[rets < 0]
    downside = neg.std() if len(neg) > 0 else rets.std()
    sortino = rets.mean() / downside * np.sqrt(252) if downside > 0 else 0

    wr = np.sum(rets > 0) / len(rets) * 100

    gains = rets[rets > 0].sum()
    losses = abs(rets[rets <= 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'CAGR': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'n_days': len(rets),
        'n_years': round(n_years, 1),
        'final_equity': round(equity[-1], 2),
        'cum_return': round((equity[-1] / initial_capital - 1) * 100, 2),
    }


def regime_test_daily(daily_returns):
    """R1 regime test."""
    green_rets = []
    red_rets = []

    for dt, ret in daily_returns.items():
        ym = dt.strftime('%Y-%m')
        if ym in regime_map:
            if regime_map[ym] == 'green':
                green_rets.append(ret)
            else:
                red_rets.append(ret)

    green_rets = np.array(green_rets)
    red_rets = np.array(red_rets)

    if len(green_rets) < 20 or len(red_rets) < 20:
        return None, None, None, False

    sg = green_rets.mean() / green_rets.std() * np.sqrt(252) if green_rets.std() > 0 else 0
    sr = red_rets.mean() / red_rets.std() * np.sqrt(252) if red_rets.std() > 0 else 0

    max_abs = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / max_abs if max_abs > 0 else 0

    return sg, sr, gap, gap <= 0.50


def permutation_test_daily(daily_returns, n_perms=100):
    """Permutation test."""
    rets = daily_returns.dropna().values
    real_sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

    count = 0
    for _ in range(n_perms):
        shuf = np.random.permutation(rets)
        s = shuf.mean() / shuf.std() * np.sqrt(252) if shuf.std() > 0 else 0
        if s >= real_sharpe:
            count += 1

    return count / n_perms


# Test configurations
print("\n[2/5] Running risk-on/risk-off variants...")

signal_sets = {
    'MA200': ['spy_200ma'],
    'MA50': ['spy_50ma'],
    'VIX': ['vix_signal', 'vix_trend'],
    'Momentum': ['spy_mom_1m', 'spy_mom_3m'],
    'DualMom': ['dual_mom'],
    'Composite3': ['spy_200ma', 'vix_signal', 'spy_mom_1m'],
    'Composite5': ['spy_200ma', 'spy_50ma', 'vix_signal', 'spy_mom_1m', 'dual_mom'],
    'Composite7': ['spy_200ma', 'spy_50ma', 'vix_signal', 'vix_trend', 'spy_mom_1m', 'spy_mom_3m', 'dual_mom'],
}

configs = []
for sig_name, sig_cols in signal_sets.items():
    # TQQQ / SHY
    configs.append({
        'signal': sig_name, 'sig_cols': sig_cols,
        'risk_on': 'TQQQ', 'risk_off': 'SHY',
        'name': f'TQQQ_{sig_name}_SHY'
    })

# Also test UPRO and with TLT as risk-off
for sig_name in ['Composite5', 'Composite7']:
    configs.append({
        'signal': sig_name, 'sig_cols': signal_sets[sig_name],
        'risk_on': 'UPRO', 'risk_off': 'SHY',
        'name': f'UPRO_{sig_name}_SHY'
    })
    configs.append({
        'signal': sig_name, 'sig_cols': signal_sets[sig_name],
        'risk_on': 'TQQQ', 'risk_off': 'TLT',
        'name': f'TQQQ_{sig_name}_TLT'
    })
    configs.append({
        'signal': sig_name, 'sig_cols': signal_sets[sig_name],
        'risk_on': 'TQQQ', 'risk_off': 'GLD',
        'name': f'TQQQ_{sig_name}_GLD'
    })

results = []
for cfg in configs:
    print(f"\n  Testing {cfg['name']}...")
    result = risk_on_off_backtest(
        close, cfg['sig_cols'],
        risk_on_ticker=cfg['risk_on'],
        risk_off_ticker=cfg['risk_off']
    )

    if result is None:
        print(f"    No data")
        continue

    daily_rets, positions = result

    if len(daily_rets.dropna()) < 252:
        print(f"    Insufficient data ({len(daily_rets.dropna())} days)")
        continue

    metrics = compute_metrics_daily(daily_rets)
    if metrics is None:
        continue

    sg, sr, gap, r1_pass = regime_test_daily(daily_rets)
    perm_p = permutation_test_daily(daily_rets)

    # Position analysis
    risk_on_pct = (positions == cfg['risk_on']).mean() * 100

    r = {
        'strategy': cfg['name'],
        **metrics,
        'risk_on_pct': round(risk_on_pct, 1),
        'sharpe_green': round(sg, 3) if sg is not None else None,
        'sharpe_red': round(sr, 3) if sr is not None else None,
        'R1_regime_gap': round(gap, 3) if gap is not None else None,
        'R1_pass': str(r1_pass),
        'perm_p_value': round(perm_p, 3),
        'perm_pass': str(perm_p < 0.05),
    }
    results.append(r)

    print(f"    Sharpe={metrics['sharpe']:.3f}, CAGR={metrics['CAGR']:.1f}%, WR={metrics['win_rate']:.1f}%")
    print(f"    MaxDD={metrics['max_drawdown']:.1f}%, Risk-ON={risk_on_pct:.0f}% of time")
    if gap is not None:
        print(f"    R1: gap={gap:.3f} ({'PASS' if r1_pass else 'FAIL'}), Green={sg:.3f}, Red={sr:.3f}")
    print(f"    Perm p={perm_p:.3f} ({'PASS' if perm_p < 0.05 else 'FAIL'})")

# Save results
print("\n[5/5] Saving results...")
with open(os.path.join(OUTPUT_DIR, 'r2_strategy3_risk_on_off.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}r2_strategy3_risk_on_off.json")
print("\n" + "=" * 70)
print("RISK-ON/RISK-OFF SUMMARY")
print("=" * 70)
for r in results:
    flags = []
    if r['R1_pass'] == 'True': flags.append('R1-PASS')
    else: flags.append('R1-FAIL')
    if r['perm_pass'] == 'True': flags.append('PERM-PASS')
    else: flags.append('PERM-FAIL')
    print(f"  {r['strategy']}: Sharpe={r['sharpe']:.3f}, CAGR={r['CAGR']:.1f}%, "
          f"WR={r['win_rate']:.0f}%, MaxDD={r['max_drawdown']:.1f}%, ON={r['risk_on_pct']:.0f}%, "
          f"[{', '.join(flags)}]")
