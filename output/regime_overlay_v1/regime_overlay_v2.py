#!/usr/bin/env python3
"""
VIX-Based Multi-Signal Regime Overlay v2
=========================================
Improvements over v1:
  1. Risk-off allocation split: SHY (safe) instead of IEF (lost money in 2022)
  2. Signal smoothing: 5-day EMA on composite score to reduce whipsaws
  3. Hysteresis: require score to cross threshold by margin to flip regime
  4. VIX momentum signal: 5d VIX change as additional input
  5. Comparison: IEF vs SHY as risk-off vehicle

All signals remain T-1 (honest).
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/regime_overlay_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COST_BPS = 5
START_DATE = '2009-01-01'
BACKTEST_START = '2010-01-01'
END_DATE = '2026-07-18'

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']


def download_data():
    """Download all required price data."""
    print("Downloading price data...")
    tickers = ['SPY', 'UPRO', '^VIX', '^VIX3M', 'HYG', 'LQD', 'IEF', 'SHY', 'TLT', 'BIL'] + SECTOR_ETFS

    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df['Close'] if 'Close' in df.columns else df['Adj Close']
            print(f"  {ticker}: {len(data[ticker])} days")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    return prices


def compute_signals_v2(prices, use_lag=True, smooth_window=5, hysteresis=0.3):
    """
    v2 signals with smoothing and hysteresis.

    Each signal produces a continuous score [0, 1] instead of binary.
    Composite is smoothed with EMA. Hysteresis prevents regime flipping on noise.
    """
    lag = 1 if use_lag else 0
    signals = pd.DataFrame(index=prices.index)

    # ── Signal 1: VIX Level (continuous) ──
    vix = prices['^VIX'].shift(lag) if lag > 0 else prices['^VIX']
    # Continuous: 1.0 at VIX=10, 0.5 at VIX=20, 0.0 at VIX=35+
    vix_score = np.clip(1.0 - (vix - 10) / 25, 0, 1)
    signals['vix_score'] = vix_score
    signals['vix_level'] = vix

    # ── Signal 2: VIX Term Structure (continuous) ──
    vix3m = prices['^VIX3M'].shift(lag) if lag > 0 else prices['^VIX3M']
    vix_ratio = vix / vix3m
    # Continuous: 1.0 at ratio=0.8, 0.5 at ratio=0.95, 0.0 at ratio=1.1+
    ts_score = np.clip(1.0 - (vix_ratio - 0.8) / 0.3, 0, 1)
    signals['ts_score'] = ts_score
    signals['vix_ratio'] = vix_ratio

    # ── Signal 3: Credit Spread (HYG/LQD 21d change) ──
    hyg_lqd = prices['HYG'] / prices['LQD']
    hyg_lqd_change = hyg_lqd.pct_change(21)
    if lag > 0:
        hyg_lqd_change = hyg_lqd_change.shift(lag)
    # Continuous: map from [-2%, +2%] to [0, 1]
    credit_score = np.clip((hyg_lqd_change + 0.02) / 0.04, 0, 1)
    signals['credit_score'] = credit_score
    signals['credit_change'] = hyg_lqd_change

    # ── Signal 4: Market Breadth ──
    available_sectors = [s for s in SECTOR_ETFS if s in prices.columns]
    breadth_scores = pd.DataFrame(index=prices.index)
    for sector in available_sectors:
        sma50 = prices[sector].rolling(50).mean()
        breadth_scores[sector] = (prices[sector] > sma50).astype(float)
    breadth_pct = breadth_scores.mean(axis=1) * 100
    if lag > 0:
        breadth_pct = breadth_pct.shift(lag)
    # Continuous: map [20%, 80%] to [0, 1]
    breadth_score = np.clip((breadth_pct - 20) / 60, 0, 1)
    signals['breadth_score'] = breadth_score
    signals['breadth_pct'] = breadth_pct

    # ── Signal 5: SPY Trend ──
    spy_sma200 = prices['SPY'].rolling(200).mean()
    spy_pct_above = (prices['SPY'] / spy_sma200 - 1) * 100  # % above/below 200 SMA
    if lag > 0:
        spy_pct_above = spy_pct_above.shift(lag)
    # Continuous: map [-10%, +10%] to [0, 1]
    trend_score = np.clip((spy_pct_above + 10) / 20, 0, 1)
    signals['trend_score'] = trend_score
    signals['spy_sma200'] = spy_sma200

    # ── Composite Score (0-5 scale) ──
    score_cols = ['vix_score', 'ts_score', 'credit_score', 'breadth_score', 'trend_score']
    signals['raw_composite'] = signals[score_cols].sum(axis=1)

    # Smooth with EMA
    signals['composite'] = signals['raw_composite'].ewm(span=smooth_window, adjust=False).mean()

    # ── Regime with Hysteresis ──
    regime = pd.Series('neutral', index=signals.index)
    current_regime = 'neutral'

    composite = signals['composite'].values
    for i in range(len(composite)):
        if np.isnan(composite[i]):
            regime.iloc[i] = np.nan
            continue

        if current_regime == 'risk_on':
            if composite[i] < 3.5 - hysteresis:  # Need to drop below 3.2 to exit risk_on
                if composite[i] < 1.5 + hysteresis:  # Drop far enough for risk_off
                    current_regime = 'risk_off'
                else:
                    current_regime = 'neutral'
        elif current_regime == 'neutral':
            if composite[i] > 3.5 + hysteresis:  # Need to rise above 3.8 for risk_on
                current_regime = 'risk_on'
            elif composite[i] < 1.5 - hysteresis:  # Drop below 1.2 for risk_off
                current_regime = 'risk_off'
        elif current_regime == 'risk_off':
            if composite[i] > 1.5 + hysteresis:  # Rise above 1.8 to exit risk_off
                if composite[i] > 3.5 + hysteresis:
                    current_regime = 'risk_on'
                else:
                    current_regime = 'neutral'

        regime.iloc[i] = current_regime

    signals['regime'] = regime

    # Exposure mapping
    exposure_map = {'risk_on': 1.0, 'neutral': 0.75, 'risk_off': 0.25}
    signals['equity_exposure'] = signals['regime'].map(exposure_map)
    signals['bond_exposure'] = 1.0 - signals['equity_exposure']

    return signals


def run_backtest(prices, signals, equity_ticker='SPY', bond_ticker='SHY', label='SPY'):
    """Run overlay backtest with transaction costs."""
    common_idx = prices.index.intersection(signals.index)
    common_idx = common_idx[common_idx >= BACKTEST_START]

    equity_ret = prices[equity_ticker].pct_change().reindex(common_idx)
    bond_ret = prices[bond_ticker].pct_change().reindex(common_idx)

    eq_exposure = signals['equity_exposure'].reindex(common_idx)
    bd_exposure = signals['bond_exposure'].reindex(common_idx)
    regime = signals['regime'].reindex(common_idx)

    valid = equity_ret.notna() & bond_ret.notna() & eq_exposure.notna()
    equity_ret = equity_ret[valid]
    bond_ret = bond_ret[valid]
    eq_exposure = eq_exposure[valid]
    bd_exposure = bd_exposure[valid]
    regime = regime[valid]

    # Transaction costs
    eq_exposure_change = eq_exposure.diff().abs().fillna(0)
    rebal_cost = eq_exposure_change * (COST_BPS / 10000) * 2

    port_ret = eq_exposure * equity_ret + bd_exposure * bond_ret - rebal_cost
    port_cum = (1 + port_ret).cumprod()
    bh_cum = (1 + equity_ret).cumprod()

    return {
        'label': label,
        'dates': port_ret.index,
        'portfolio_returns': port_ret,
        'portfolio_equity_curve': port_cum,
        'buyhold_returns': equity_ret,
        'buyhold_equity_curve': bh_cum,
        'regime': regime,
        'equity_exposure': eq_exposure,
        'rebalance_cost_total': rebal_cost.sum(),
        'num_rebalances': (eq_exposure_change > 0).sum(),
    }


def compute_metrics(returns, label=''):
    """Compute risk-adjusted performance metrics."""
    returns = returns.dropna()
    if len(returns) < 30:
        return {'label': label, 'error': 'insufficient data'}

    total_ret = (1 + returns).prod() - 1
    years = len(returns) / 252
    cagr = (1 + total_ret) ** (1/years) - 1

    ann_vol = returns.std() * np.sqrt(252)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = cagr / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wr = (returns > 0).mean()
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'label': label,
        'total_return': f"{total_ret:.1%}",
        'cagr': f"{cagr:.1%}",
        'ann_vol': f"{ann_vol:.1%}",
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'max_dd': f"{max_dd:.1%}",
        'calmar': round(calmar, 2),
        'win_rate': f"{wr:.1%}",
        'profit_factor': round(pf, 2),
        'years': round(years, 1),
    }


def compute_regime_metrics(results):
    """Performance by regime."""
    regime = results['regime']
    port_ret = results['portfolio_returns']

    stats = {}
    for r in ['risk_on', 'neutral', 'risk_off']:
        mask = regime == r
        if mask.sum() < 10:
            continue
        rets = port_ret[mask]
        pct_time = mask.mean()
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        stats[r] = {
            'pct_time': f"{pct_time:.1%}",
            'days': int(mask.sum()),
            'ann_return': f"{ann_ret:.1%}",
            'ann_vol': f"{ann_vol:.1%}",
            'sharpe': round(sharpe, 2),
        }
    return stats


def plot_comparison(all_results, filename='v2_comparison.png'):
    """Plot v2 results comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # SPY equity curves
    ax = axes[0, 0]
    for r in all_results:
        if 'SPY' in r['label']:
            if 'B&H' in r['label']:
                ax.plot(r['dates'], r['buyhold_equity_curve'], label=r['label'],
                        linewidth=1.2, alpha=0.7, color='blue')
            else:
                color = 'green' if 'SHY' in r['label'] else 'orange' if 'IEF' in r['label'] else 'red'
                ax.plot(r['dates'], r['portfolio_equity_curve'], label=r['label'],
                        linewidth=1.5, alpha=0.8, color=color)
    ax.set_yscale('log')
    ax.set_title('SPY: Buy & Hold vs Regime Overlay v2', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # UPRO equity curves
    ax = axes[0, 1]
    for r in all_results:
        if 'UPRO' in r['label']:
            if 'B&H' in r['label']:
                ax.plot(r['dates'], r['buyhold_equity_curve'], label=r['label'],
                        linewidth=1.2, alpha=0.7, color='blue')
            else:
                color = 'green' if 'SHY' in r['label'] else 'orange' if 'IEF' in r['label'] else 'red'
                ax.plot(r['dates'], r['portfolio_equity_curve'], label=r['label'],
                        linewidth=1.5, alpha=0.8, color=color)
    ax.set_yscale('log')
    ax.set_title('UPRO: Buy & Hold vs Regime Overlay v2', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # SPY drawdowns
    ax = axes[1, 0]
    for r in all_results:
        if 'SPY' in r['label']:
            if 'B&H' in r['label']:
                cum = r['buyhold_equity_curve']
            else:
                cum = r['portfolio_equity_curve']
            dd = (cum - cum.cummax()) / cum.cummax() * 100
            color = 'blue' if 'B&H' in r['label'] else ('green' if 'SHY' in r['label'] else 'orange')
            ax.plot(r['dates'], dd, label=r['label'], linewidth=0.8, alpha=0.7, color=color)
    ax.set_title('SPY Drawdowns', fontsize=13, fontweight='bold')
    ax.set_ylabel('Drawdown %')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # UPRO drawdowns
    ax = axes[1, 1]
    for r in all_results:
        if 'UPRO' in r['label']:
            if 'B&H' in r['label']:
                cum = r['buyhold_equity_curve']
            else:
                cum = r['portfolio_equity_curve']
            dd = (cum - cum.cummax()) / cum.cummax() * 100
            color = 'blue' if 'B&H' in r['label'] else ('green' if 'SHY' in r['label'] else 'orange')
            ax.plot(r['dates'], dd, label=r['label'], linewidth=0.8, alpha=0.7, color=color)
    ax.set_title('UPRO Drawdowns', fontsize=13, fontweight='bold')
    ax.set_ylabel('Drawdown %')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_composite_score(signals, filename='v2_composite_score.png'):
    """Plot composite score and regime."""
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

    idx = signals.index[signals.index >= BACKTEST_START]
    s = signals.reindex(idx)

    # Panel 1: Individual scores
    ax = axes[0]
    for col, label, color in [('vix_score', 'VIX', 'purple'),
                               ('ts_score', 'Term Struct', 'teal'),
                               ('credit_score', 'Credit', 'brown'),
                               ('breadth_score', 'Breadth', 'navy'),
                               ('trend_score', 'Trend', 'darkgreen')]:
        ax.plot(idx, s[col], label=label, alpha=0.6, linewidth=0.7, color=color)
    ax.set_ylabel('Signal Score (0-1)')
    ax.set_title('Individual Signal Scores (v2 continuous)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', ncol=5, fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Composite with thresholds
    ax = axes[1]
    ax.plot(idx, s['raw_composite'], label='Raw', alpha=0.3, linewidth=0.5, color='gray')
    ax.plot(idx, s['composite'], label='Smoothed (EMA-5)', alpha=0.8, linewidth=1.2, color='blue')
    ax.axhline(3.5, color='green', linestyle='--', alpha=0.5, label='Risk-On threshold')
    ax.axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Risk-Off threshold')
    ax.set_ylabel('Composite Score (0-5)')
    ax.set_title('Composite Score with Hysteresis Thresholds', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Regime & exposure
    ax = axes[2]
    regime_colors = s['regime'].map({'risk_on': 'green', 'neutral': 'gold', 'risk_off': 'red'})
    ax.scatter(idx, s['equity_exposure'] * 100, c=regime_colors, s=2, alpha=0.5)
    ax.set_ylabel('Equity Exposure %')
    ax.set_ylim(0, 110)
    ax.set_title('Equity Exposure by Regime', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def yearly_returns_table(results, label):
    """Compute yearly returns comparison."""
    port_ret = results['portfolio_returns']
    bh_ret = results['buyhold_returns']

    years = sorted(port_ret.index.year.unique())
    rows = []
    for y in years:
        mask = port_ret.index.year == y
        if mask.sum() < 20:
            continue
        p_ret = (1 + port_ret[mask]).prod() - 1
        b_ret = (1 + bh_ret[mask]).prod() - 1
        excess = p_ret - b_ret
        rows.append({'year': y, 'buyhold': b_ret, 'overlay': p_ret, 'excess': excess})

    return pd.DataFrame(rows)


def main():
    print("="*70)
    print("VIX-Based Multi-Signal Regime Overlay v2")
    print("="*70)

    # Try to load cached prices
    cache_path = os.path.join(OUTPUT_DIR, 'raw_prices.parquet')
    if os.path.exists(cache_path):
        print("Loading cached price data...")
        prices = pd.read_parquet(cache_path)
        # Need SHY and BIL - download if missing
        missing = [t for t in ['SHY', 'BIL', 'TLT'] if t not in prices.columns]
        if missing:
            print(f"Downloading missing tickers: {missing}")
            for ticker in missing:
                try:
                    df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    prices[ticker] = df['Close'] if 'Close' in df.columns else df['Adj Close']
                except:
                    pass
            prices.to_parquet(cache_path)
    else:
        prices = download_data()
        prices.to_parquet(cache_path)

    # Compute v2 signals
    print("\nComputing v2 signals (continuous + smoothed + hysteresis, T-1)...")
    signals = compute_signals_v2(prices, use_lag=True)

    # Regime distribution
    bt_signals = signals[signals.index >= BACKTEST_START].dropna(subset=['composite'])
    regime_dist = bt_signals['regime'].value_counts(normalize=True)
    regime_counts = bt_signals['regime'].value_counts()

    print("\n" + "="*70)
    print("v2 REGIME DISTRIBUTION")
    print("="*70)
    for r in ['risk_on', 'neutral', 'risk_off']:
        if r in regime_dist:
            print(f"  {r:<12}: {regime_dist[r]:.1%} ({regime_counts[r]} days)")

    # ── Run Backtests ──
    print("\n" + "="*70)
    print("v2 BACKTEST RESULTS (T-1, 5 bps, SHY as risk-off vehicle)")
    print("="*70)

    # SPY with SHY
    spy_shy = run_backtest(prices, signals, 'SPY', 'SHY', 'SPY+Overlay(SHY)')
    # SPY with IEF (for comparison)
    spy_ief = run_backtest(prices, signals, 'SPY', 'IEF', 'SPY+Overlay(IEF)')
    # SPY B&H (dummy for labeling)
    spy_bh_results = run_backtest(prices, signals, 'SPY', 'SHY', 'SPY B&H')

    # UPRO with SHY
    upro_shy = run_backtest(prices, signals, 'UPRO', 'SHY', 'UPRO+Overlay(SHY)')
    # UPRO with IEF
    upro_ief = run_backtest(prices, signals, 'UPRO', 'IEF', 'UPRO+Overlay(IEF)')
    upro_bh_results = run_backtest(prices, signals, 'UPRO', 'SHY', 'UPRO B&H')

    all_metrics = [
        compute_metrics(spy_bh_results['buyhold_returns'], 'SPY B&H'),
        compute_metrics(spy_shy['portfolio_returns'], 'SPY+Overlay(SHY)'),
        compute_metrics(spy_ief['portfolio_returns'], 'SPY+Overlay(IEF)'),
        compute_metrics(upro_bh_results['buyhold_returns'], 'UPRO B&H'),
        compute_metrics(upro_shy['portfolio_returns'], 'UPRO+Overlay(SHY)'),
        compute_metrics(upro_ief['portfolio_returns'], 'UPRO+Overlay(IEF)'),
    ]

    print(f"\n{'Strategy':<28} {'CAGR':<10} {'Sharpe':<10} {'Sortino':<10} {'MaxDD':<12} {'Calmar':<10}")
    print("-" * 80)
    for m in all_metrics:
        print(f"{m['label']:<28} {m['cagr']:<10} {m['sharpe']:<10} {m['sortino']:<10} "
              f"{m['max_dd']:<12} {m['calmar']:<10}")

    # Regime performance
    print("\n" + "="*70)
    print("v2 PERFORMANCE BY REGIME")
    print("="*70)
    for results in [spy_shy, upro_shy]:
        regime_stats = compute_regime_metrics(results)
        print(f"\n{results['label']}:")
        print(f"  {'Regime':<12} {'% Time':<10} {'Days':<8} {'Ann Ret':<12} {'Sharpe':<10}")
        print("  " + "-"*50)
        for r in ['risk_on', 'neutral', 'risk_off']:
            if r in regime_stats:
                s = regime_stats[r]
                print(f"  {r:<12} {s['pct_time']:<10} {s['days']:<8} {s['ann_return']:<12} {s['sharpe']:<10}")

    # Transaction costs
    print(f"\n  SPY rebalances: {spy_shy['num_rebalances']}, cost: {spy_shy['rebalance_cost_total']:.4f}")
    print(f"  UPRO rebalances: {upro_shy['num_rebalances']}, cost: {upro_shy['rebalance_cost_total']:.4f}")

    # ── Crisis Analysis ──
    print("\n" + "="*70)
    print("v2 CRISIS PERIOD ANALYSIS")
    print("="*70)

    crisis_periods = [
        ('2011 EU Debt Crisis', '2011-07-01', '2011-10-31'),
        ('2015-16 China Deval', '2015-08-01', '2016-02-29'),
        ('2018 Q4 Selloff', '2018-10-01', '2018-12-31'),
        ('COVID Crash', '2020-02-01', '2020-04-30'),
        ('2022 Bear Market', '2022-01-01', '2022-12-31'),
        ('2020 Recovery', '2020-04-01', '2020-12-31'),
        ('2023 Recovery', '2023-01-01', '2023-12-31'),
    ]

    crisis_data = {}
    for crisis_name, start, end in crisis_periods:
        mask = (spy_shy['dates'] >= start) & (spy_shy['dates'] <= end)
        if mask.sum() == 0:
            continue

        spy_bh_ret = float((1 + spy_shy['buyhold_returns'][mask]).prod() - 1)
        spy_ov_ret = float((1 + spy_shy['portfolio_returns'][mask]).prod() - 1)
        upro_bh_ret = float((1 + upro_shy['buyhold_returns'][mask]).prod() - 1)
        upro_ov_ret = float((1 + upro_shy['portfolio_returns'][mask]).prod() - 1)

        crisis_regime = spy_shy['regime'][mask]
        regime_str = ', '.join([f"{r}: {v:.0%}" for r, v in crisis_regime.value_counts(normalize=True).items()])

        print(f"\n  {crisis_name} ({start} to {end}):")
        print(f"    SPY  B&H: {spy_bh_ret:+.1%}  | Overlay: {spy_ov_ret:+.1%}  | Delta: {spy_ov_ret - spy_bh_ret:+.1%}")
        print(f"    UPRO B&H: {upro_bh_ret:+.1%}  | Overlay: {upro_ov_ret:+.1%}  | Delta: {upro_ov_ret - upro_bh_ret:+.1%}")
        print(f"    Regime: {regime_str}")

        crisis_data[crisis_name] = {
            'SPY_bh': f"{spy_bh_ret:+.1%}", 'SPY_overlay': f"{spy_ov_ret:+.1%}",
            'UPRO_bh': f"{upro_bh_ret:+.1%}", 'UPRO_overlay': f"{upro_ov_ret:+.1%}",
        }

    # ── Lookahead Validation ──
    print("\n" + "="*70)
    print("v2 LOOKAHEAD VALIDATION: T-0 vs T-1")
    print("="*70)

    signals_t0 = compute_signals_v2(prices, use_lag=False)
    spy_t0 = run_backtest(prices, signals_t0, 'SPY', 'SHY', 'SPY T-0')
    m_t0 = compute_metrics(spy_t0['portfolio_returns'], 'T-0')
    m_t1 = compute_metrics(spy_shy['portfolio_returns'], 'T-1')

    print(f"\n{'Metric':<15} {'T-0 (lookahead)':<20} {'T-1 (honest)':<20}")
    print("-"*55)
    for key in ['cagr', 'sharpe', 'sortino', 'max_dd']:
        print(f"{key:<15} {m_t0[key]:<20} {m_t1[key]:<20}")

    sharpe_gap = m_t0['sharpe'] - m_t1['sharpe']
    print(f"\nSharpe gap: {sharpe_gap:.2f}")

    # ── Yearly Returns ──
    print("\n" + "="*70)
    print("v2 YEARLY RETURNS")
    print("="*70)

    for results, label in [(spy_shy, 'SPY'), (upro_shy, 'UPRO')]:
        yearly = yearly_returns_table(results, label)
        print(f"\n{label}:")
        print(f"  {'Year':<8} {'B&H':<12} {'Overlay':<12} {'Excess':<12} {'Winner':<10}")
        print("  " + "-"*52)
        wins = 0
        for _, row in yearly.iterrows():
            winner = 'OVERLAY' if row['excess'] > 0 else 'B&H'
            if row['excess'] > 0:
                wins += 1
            print(f"  {int(row['year']):<8} {row['buyhold']:+.1%}{'':>4} {row['overlay']:+.1%}{'':>4} "
                  f"{row['excess']:+.1%}{'':>4} {winner:<10}")
        total_years = len(yearly)
        print(f"  Overlay wins: {wins}/{total_years} years ({wins/total_years:.0%})")

    # ── Plots ──
    print("\nGenerating plots...")
    all_plot_results = [
        {'label': 'SPY B&H', 'dates': spy_shy['dates'],
         'buyhold_equity_curve': spy_shy['buyhold_equity_curve'],
         'portfolio_equity_curve': spy_shy['buyhold_equity_curve']},
        spy_shy, spy_ief,
        {'label': 'UPRO B&H', 'dates': upro_shy['dates'],
         'buyhold_equity_curve': upro_shy['buyhold_equity_curve'],
         'portfolio_equity_curve': upro_shy['buyhold_equity_curve']},
        upro_shy, upro_ief,
    ]
    plot_comparison(all_plot_results)
    plot_composite_score(signals)

    # ── Save Results ──
    summary = {
        'version': 'v2',
        'improvements': [
            'Continuous signal scores (0-1) instead of binary',
            'EMA-5 smoothing on composite score',
            'Hysteresis (0.3) to reduce regime flip whipsaws',
            'SHY as risk-off vehicle (avoids 2022 IEF loss)',
        ],
        'backtest_period': f"{BACKTEST_START} to {END_DATE}",
        'signal_lag': 'T-1 (honest)',
        'transaction_cost_bps': COST_BPS,
        'regime_distribution': {r: f"{regime_dist.get(r, 0):.1%}" for r in ['risk_on', 'neutral', 'risk_off']},
        'metrics': {m['label']: m for m in all_metrics},
        'lookahead_validation': {
            'T0_sharpe': m_t0['sharpe'],
            'T1_sharpe': m_t1['sharpe'],
            'sharpe_gap': round(sharpe_gap, 2),
        },
        'crisis_protection': crisis_data,
    }

    with open(os.path.join(OUTPUT_DIR, 'v2_backtest_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    signals.to_parquet(os.path.join(OUTPUT_DIR, 'v2_regime_signals.parquet'))

    print(f"\nAll v2 results saved to {OUTPUT_DIR}/")
    print("\nDONE.")


if __name__ == '__main__':
    main()
