#!/usr/bin/env python3
"""
VIX-Based Multi-Signal Regime Overlay v1
=========================================
Rules-based regime detector using 5 signals (all T-1):
  1. VIX Level
  2. VIX Term Structure (VIX/VIX3M)
  3. Credit Spread (HYG/LQD 21d change)
  4. Market Breadth (% S&P sectors above 50d SMA)
  5. SPY Trend (above/below 200d SMA)

Regime Classification:
  Risk-On  (4-5 bullish): 100% equity exposure
  Neutral  (2-3 bullish): 75% equity exposure
  Risk-Off (0-1 bullish): 25% equity, 75% IEF

Backtest: 2010-01-01 to 2026-07-18
Applies overlay to SPY (buy-and-hold) and UPRO (3x leveraged S&P)
Transaction costs: 5 bps per rebalance leg
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
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/regime_overlay_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

COST_BPS = 5  # 5 bps per leg
START_DATE = '2009-01-01'  # extra lookback for 200d SMA
BACKTEST_START = '2010-01-01'
END_DATE = '2026-07-18'

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']

# ─── DATA DOWNLOAD ───────────────────────────────────────────────────────────

def download_data():
    """Download all required price data."""
    print("Downloading price data...")

    # Core tickers
    tickers = ['SPY', 'UPRO', '^VIX', '^VIX3M', 'HYG', 'LQD', 'IEF'] + SECTOR_ETFS

    data = {}
    for ticker in tickers:
        print(f"  {ticker}...", end=' ')
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[ticker] = df['Close'] if 'Close' in df.columns else df['Adj Close']
            print(f"{len(data[ticker])} days")
        except Exception as e:
            print(f"FAILED: {e}")

    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    # Remove timezone if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    print(f"\nTotal trading days: {len(prices)}")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Missing data summary:")
    for col in prices.columns:
        missing = prices[col].isna().sum()
        if missing > 0:
            first_valid = prices[col].first_valid_index()
            print(f"  {col}: {missing} missing, starts {first_valid.date() if first_valid else 'N/A'}")

    return prices


# ─── SIGNAL GENERATION ────────────────────────────────────────────────────────

def compute_signals(prices, use_lag=True):
    """
    Compute 5 regime signals. If use_lag=True, signals use T-1 data (honest).
    If use_lag=False, signals use T-0 data (for lookahead comparison).
    """
    lag = 1 if use_lag else 0

    signals = pd.DataFrame(index=prices.index)

    # ── Signal 1: VIX Level ──
    # <15 complacent (bullish), 15-20 normal (bullish), 20-30 elevated (bearish), >30 crisis (bearish)
    vix = prices['^VIX'].shift(lag) if lag > 0 else prices['^VIX']
    signals['vix_bullish'] = (vix < 20).astype(float)
    signals['vix_level'] = vix

    # ── Signal 2: VIX Term Structure ──
    # VIX/VIX3M: <0.85 deep contango (bullish), 0.85-1.0 mild contango (bullish), >1.0 backwardation (bearish)
    vix3m = prices['^VIX3M'].shift(lag) if lag > 0 else prices['^VIX3M']
    vix_ratio = vix / vix3m
    signals['term_structure_bullish'] = (vix_ratio < 1.0).astype(float)
    signals['vix_ratio'] = vix_ratio

    # ── Signal 3: Credit Spread (HYG/LQD ratio, 21d change) ──
    # Rising ratio = credit tightening = bullish
    hyg_lqd = (prices['HYG'] / prices['LQD'])
    hyg_lqd_change = hyg_lqd.pct_change(21)
    if lag > 0:
        hyg_lqd_change = hyg_lqd_change.shift(lag)
    signals['credit_bullish'] = (hyg_lqd_change > 0).astype(float)
    signals['credit_change'] = hyg_lqd_change

    # ── Signal 4: Market Breadth ──
    # % of S&P sectors above their 50d SMA. >70% = bullish, <30% = bearish
    available_sectors = [s for s in SECTOR_ETFS if s in prices.columns]
    breadth_scores = pd.DataFrame(index=prices.index)
    for sector in available_sectors:
        sma50 = prices[sector].rolling(50).mean()
        breadth_scores[sector] = (prices[sector] > sma50).astype(float)

    breadth_pct = breadth_scores.mean(axis=1) * 100
    if lag > 0:
        breadth_pct = breadth_pct.shift(lag)
    signals['breadth_bullish'] = (breadth_pct > 50).astype(float)  # >50% as threshold for bullish
    signals['breadth_pct'] = breadth_pct

    # ── Signal 5: SPY Trend ──
    # SPY above 200d SMA = bullish
    spy_sma200 = prices['SPY'].rolling(200).mean()
    spy_above_200 = prices['SPY'] > spy_sma200
    if lag > 0:
        spy_above_200 = spy_above_200.shift(lag)
    signals['trend_bullish'] = spy_above_200.astype(float)
    signals['spy_sma200'] = spy_sma200

    # ── Composite Score ──
    bullish_cols = ['vix_bullish', 'term_structure_bullish', 'credit_bullish',
                    'breadth_bullish', 'trend_bullish']
    signals['bullish_count'] = signals[bullish_cols].sum(axis=1)

    # ── Regime Classification ──
    signals['regime'] = 'neutral'
    signals.loc[signals['bullish_count'] >= 4, 'regime'] = 'risk_on'
    signals.loc[signals['bullish_count'] <= 1, 'regime'] = 'risk_off'

    # Exposure mapping
    exposure_map = {'risk_on': 1.0, 'neutral': 0.75, 'risk_off': 0.25}
    signals['equity_exposure'] = signals['regime'].map(exposure_map)
    signals['bond_exposure'] = 1.0 - signals['equity_exposure']

    return signals


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────

def run_backtest(prices, signals, equity_ticker='SPY', bond_ticker='IEF', label='SPY'):
    """
    Run overlay backtest with transaction costs.

    equity_exposure allocated to equity_ticker
    bond_exposure allocated to bond_ticker (IEF)
    Transaction cost: COST_BPS per leg per rebalance
    """
    # Align dates
    common_idx = prices.index.intersection(signals.index)
    common_idx = common_idx[common_idx >= BACKTEST_START]

    equity_ret = prices[equity_ticker].pct_change().reindex(common_idx)
    bond_ret = prices[bond_ticker].pct_change().reindex(common_idx)

    eq_exposure = signals['equity_exposure'].reindex(common_idx)
    bd_exposure = signals['bond_exposure'].reindex(common_idx)
    regime = signals['regime'].reindex(common_idx)

    # Drop NaN rows
    valid = equity_ret.notna() & bond_ret.notna() & eq_exposure.notna()
    equity_ret = equity_ret[valid]
    bond_ret = bond_ret[valid]
    eq_exposure = eq_exposure[valid]
    bd_exposure = bd_exposure[valid]
    regime = regime[valid]

    # Transaction costs on rebalance
    eq_exposure_change = eq_exposure.diff().abs().fillna(0)
    # Cost = change in exposure * cost_bps (each leg)
    # Two legs: selling equity + buying bond (or vice versa)
    rebal_cost = eq_exposure_change * (COST_BPS / 10000) * 2  # 2 legs

    # Portfolio return
    port_ret = eq_exposure * equity_ret + bd_exposure * bond_ret - rebal_cost

    # Equity curves
    port_cum = (1 + port_ret).cumprod()
    equity_cum = (1 + equity_ret).cumprod()
    bond_cum = (1 + bond_ret).cumprod()

    # Buy-and-hold baseline
    bh_ret = equity_ret.copy()
    bh_cum = (1 + bh_ret).cumprod()

    results = {
        'label': label,
        'dates': port_ret.index,
        'portfolio_returns': port_ret,
        'portfolio_equity_curve': port_cum,
        'buyhold_returns': bh_ret,
        'buyhold_equity_curve': bh_cum,
        'bond_equity_curve': bond_cum,
        'regime': regime,
        'equity_exposure': eq_exposure,
        'rebalance_cost_total': rebal_cost.sum(),
        'num_rebalances': (eq_exposure_change > 0).sum(),
    }

    return results


def compute_metrics(returns, label=''):
    """Compute risk-adjusted performance metrics."""
    if len(returns.dropna()) < 30:
        return {'label': label, 'error': 'insufficient data'}

    returns = returns.dropna()
    ann_factor = 252

    total_ret = (1 + returns).prod() - 1
    years = len(returns) / ann_factor
    cagr = (1 + total_ret) ** (1/years) - 1

    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = cagr / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(ann_factor)
    sortino = cagr / downside if downside > 0 else 0

    # Max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (returns > 0).mean()

    # Profit factor
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
    """Compute performance broken down by regime."""
    regime = results['regime']
    port_ret = results['portfolio_returns']

    regime_stats = {}
    for r in ['risk_on', 'neutral', 'risk_off']:
        mask = regime == r
        if mask.sum() < 10:
            continue
        rets = port_ret[mask]
        pct_time = mask.mean()

        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        regime_stats[r] = {
            'pct_time': f"{pct_time:.1%}",
            'days': int(mask.sum()),
            'ann_return': f"{ann_ret:.1%}",
            'ann_vol': f"{ann_vol:.1%}",
            'sharpe': round(sharpe, 2),
            'avg_daily_ret': f"{rets.mean():.4%}",
        }

    return regime_stats


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

def plot_equity_curves(results_list, filename='equity_curves.png'):
    """Plot equity curves for overlay vs buy-and-hold."""
    fig, axes = plt.subplots(len(results_list), 1, figsize=(16, 6*len(results_list)), squeeze=False)

    for i, results in enumerate(results_list):
        ax = axes[i, 0]
        label = results['label']

        dates = results['dates']
        ax.plot(dates, results['buyhold_equity_curve'], label=f'{label} Buy & Hold',
                alpha=0.7, linewidth=1.2, color='blue')
        ax.plot(dates, results['portfolio_equity_curve'], label=f'{label} + Regime Overlay',
                alpha=0.9, linewidth=1.5, color='green')

        # Shade risk-off periods
        regime = results['regime']
        risk_off_start = None
        for j, (date, r) in enumerate(zip(dates, regime)):
            if r == 'risk_off' and risk_off_start is None:
                risk_off_start = date
            elif r != 'risk_off' and risk_off_start is not None:
                ax.axvspan(risk_off_start, date, alpha=0.15, color='red', label='_')
                risk_off_start = None
        if risk_off_start is not None:
            ax.axvspan(risk_off_start, dates[-1], alpha=0.15, color='red')

        ax.set_yscale('log')
        ax.set_title(f'{label}: Buy & Hold vs Regime Overlay (T-1 Signals)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Growth of $1 (log scale)')
        ax.legend(loc='upper left', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_regime_timeline(signals, filename='regime_timeline.png'):
    """Plot regime classification over time."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    idx = signals.index[signals.index >= BACKTEST_START]
    signals_bt = signals.reindex(idx)

    # Panel 1: Regime
    regime_num = signals_bt['regime'].map({'risk_on': 2, 'neutral': 1, 'risk_off': 0})
    colors = regime_num.map({2: 'green', 1: 'gold', 0: 'red'})
    axes[0].scatter(idx, regime_num, c=colors, s=1, alpha=0.5)
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(['Risk-Off', 'Neutral', 'Risk-On'])
    axes[0].set_title('Regime Classification Over Time', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Panel 2: VIX + Term Structure
    ax2 = axes[1]
    ax2.plot(idx, signals_bt['vix_level'], label='VIX', color='purple', alpha=0.7, linewidth=0.8)
    ax2.axhline(20, color='orange', linestyle='--', alpha=0.5, label='VIX=20')
    ax2.set_ylabel('VIX Level')
    ax2.legend(loc='upper left')
    ax2r = ax2.twinx()
    ax2r.plot(idx, signals_bt['vix_ratio'], label='VIX/VIX3M', color='teal', alpha=0.7, linewidth=0.8)
    ax2r.axhline(1.0, color='red', linestyle='--', alpha=0.5)
    ax2r.set_ylabel('VIX/VIX3M')
    ax2r.legend(loc='upper right')
    ax2.set_title('VIX Level & Term Structure', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Credit + Breadth
    ax3 = axes[2]
    ax3.plot(idx, signals_bt['credit_change'] * 100, label='HYG/LQD 21d Chg %', color='brown', alpha=0.7, linewidth=0.8)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Credit Change (%)')
    ax3.legend(loc='upper left')
    ax3r = ax3.twinx()
    ax3r.plot(idx, signals_bt['breadth_pct'], label='Breadth %', color='navy', alpha=0.7, linewidth=0.8)
    ax3r.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax3r.set_ylabel('% Sectors > 50d SMA')
    ax3r.legend(loc='upper right')
    ax3.set_title('Credit Spread & Market Breadth', fontsize=12)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Equity Exposure
    axes[3].fill_between(idx, signals_bt['equity_exposure'] * 100, alpha=0.4, color='green')
    axes[3].plot(idx, signals_bt['equity_exposure'] * 100, color='green', linewidth=0.8)
    axes[3].set_ylabel('Equity Exposure %')
    axes[3].set_ylim(0, 110)
    axes[3].set_title('Portfolio Equity Exposure', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


def plot_drawdowns(results_list, filename='drawdowns.png'):
    """Compare drawdowns between overlay and buy-and-hold."""
    fig, axes = plt.subplots(len(results_list), 1, figsize=(16, 5*len(results_list)), squeeze=False)

    for i, results in enumerate(results_list):
        ax = axes[i, 0]
        label = results['label']

        # Buy-and-hold drawdown
        bh_cum = results['buyhold_equity_curve']
        bh_dd = (bh_cum - bh_cum.cummax()) / bh_cum.cummax() * 100

        # Overlay drawdown
        ov_cum = results['portfolio_equity_curve']
        ov_dd = (ov_cum - ov_cum.cummax()) / ov_cum.cummax() * 100

        ax.fill_between(results['dates'], bh_dd, alpha=0.3, color='red', label=f'{label} B&H DD')
        ax.fill_between(results['dates'], ov_dd, alpha=0.3, color='green', label=f'{label} Overlay DD')
        ax.set_ylabel('Drawdown %')
        ax.set_title(f'{label}: Drawdown Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {filename}")


# ─── LOOKAHEAD VALIDATION ────────────────────────────────────────────────────

def lookahead_comparison(prices):
    """Compare T-0 vs T-1 signals to detect lookahead bias."""
    print("\n" + "="*70)
    print("LOOKAHEAD VALIDATION: T-0 vs T-1 Signal Comparison")
    print("="*70)

    signals_t0 = compute_signals(prices, use_lag=False)
    signals_t1 = compute_signals(prices, use_lag=True)

    results_t0 = run_backtest(prices, signals_t0, 'SPY', 'IEF', 'SPY T-0')
    results_t1 = run_backtest(prices, signals_t1, 'SPY', 'IEF', 'SPY T-1')

    m_t0 = compute_metrics(results_t0['portfolio_returns'], 'SPY + Overlay (T-0, LOOKAHEAD)')
    m_t1 = compute_metrics(results_t1['portfolio_returns'], 'SPY + Overlay (T-1, HONEST)')

    print(f"\n{'Metric':<20} {'T-0 (lookahead)':<20} {'T-1 (honest)':<20}")
    print("-"*60)
    for key in ['cagr', 'sharpe', 'sortino', 'max_dd', 'calmar']:
        print(f"{key:<20} {m_t0[key]:<20} {m_t1[key]:<20}")

    sharpe_gap = abs(m_t0['sharpe'] - m_t1['sharpe'])
    print(f"\nSharpe gap (T-0 - T-1): {sharpe_gap:.2f}")
    if sharpe_gap > 0.5:
        print("WARNING: Large Sharpe gap suggests signals are partially stale at T-1.")
    else:
        print("OK: Modest gap, T-1 signals retain most of their value.")

    return m_t0, m_t1


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("="*70)
    print("VIX-Based Multi-Signal Regime Overlay Backtest v1")
    print("="*70)

    # Download data
    prices = download_data()

    # Save raw data
    prices.to_parquet(os.path.join(OUTPUT_DIR, 'raw_prices.parquet'))

    # Compute signals (T-1, honest)
    print("\nComputing regime signals (T-1 lag)...")
    signals = compute_signals(prices, use_lag=True)

    # Signal availability
    bt_signals = signals[signals.index >= BACKTEST_START]
    valid_signals = bt_signals['bullish_count'].dropna()
    print(f"Valid signal days: {len(valid_signals)} ({valid_signals.index[0].date()} to {valid_signals.index[-1].date()})")

    # Regime distribution
    print("\n" + "="*70)
    print("REGIME DISTRIBUTION")
    print("="*70)
    regime_dist = bt_signals['regime'].value_counts(normalize=True)
    regime_counts = bt_signals['regime'].value_counts()
    for r in ['risk_on', 'neutral', 'risk_off']:
        if r in regime_dist:
            print(f"  {r:<12}: {regime_dist[r]:.1%} ({regime_counts[r]} days)")

    # Bullish count distribution
    print("\nBullish signal count distribution:")
    bc_dist = bt_signals['bullish_count'].value_counts().sort_index()
    for count, days in bc_dist.items():
        pct = days / len(bt_signals)
        print(f"  {int(count)} signals bullish: {days} days ({pct:.1%})")

    # ── Run Backtests ──
    print("\n" + "="*70)
    print("BACKTEST RESULTS (T-1 Signals, 5 bps transaction cost)")
    print("="*70)

    # SPY backtest
    spy_results = run_backtest(prices, signals, 'SPY', 'IEF', 'SPY')
    spy_bh_metrics = compute_metrics(spy_results['buyhold_returns'], 'SPY Buy & Hold')
    spy_ov_metrics = compute_metrics(spy_results['portfolio_returns'], 'SPY + Regime Overlay')

    # UPRO backtest
    upro_results = run_backtest(prices, signals, 'UPRO', 'IEF', 'UPRO')
    upro_bh_metrics = compute_metrics(upro_results['buyhold_returns'], 'UPRO Buy & Hold')
    upro_ov_metrics = compute_metrics(upro_results['portfolio_returns'], 'UPRO + Regime Overlay')

    # Print results
    all_metrics = [spy_bh_metrics, spy_ov_metrics, upro_bh_metrics, upro_ov_metrics]

    print(f"\n{'Strategy':<28} {'CAGR':<10} {'Sharpe':<10} {'Sortino':<10} {'MaxDD':<12} {'Calmar':<10} {'WR':<8}")
    print("-" * 88)
    for m in all_metrics:
        print(f"{m['label']:<28} {m['cagr']:<10} {m['sharpe']:<10} {m['sortino']:<10} "
              f"{m['max_dd']:<12} {m['calmar']:<10} {m['win_rate']:<8}")

    # Regime-specific performance
    print("\n" + "="*70)
    print("PERFORMANCE BY REGIME")
    print("="*70)

    for results in [spy_results, upro_results]:
        regime_stats = compute_regime_metrics(results)
        print(f"\n{results['label']} + Overlay:")
        print(f"  {'Regime':<12} {'% Time':<10} {'Days':<8} {'Ann Ret':<12} {'Ann Vol':<12} {'Sharpe':<10}")
        print("  " + "-"*62)
        for r in ['risk_on', 'neutral', 'risk_off']:
            if r in regime_stats:
                s = regime_stats[r]
                print(f"  {r:<12} {s['pct_time']:<10} {s['days']:<8} {s['ann_return']:<12} "
                      f"{s['ann_vol']:<12} {s['sharpe']:<10}")

    # Transaction cost summary
    print("\n" + "="*70)
    print("TRANSACTION COSTS")
    print("="*70)
    for results in [spy_results, upro_results]:
        print(f"  {results['label']}: {results['num_rebalances']} rebalances, "
              f"total cost: {results['rebalance_cost_total']:.4f} ({results['rebalance_cost_total']*100:.2f}%)")

    # ── Drawdown Analysis ──
    print("\n" + "="*70)
    print("DRAWDOWN ANALYSIS - KEY CRISIS PERIODS")
    print("="*70)

    crisis_periods = [
        ('2011 EU Debt Crisis', '2011-07-01', '2011-10-31'),
        ('2015-16 China Deval', '2015-08-01', '2016-02-29'),
        ('2018 Q4 Selloff', '2018-10-01', '2018-12-31'),
        ('COVID Crash', '2020-02-01', '2020-04-30'),
        ('2022 Bear Market', '2022-01-01', '2022-12-31'),
    ]

    for crisis_name, start, end in crisis_periods:
        mask = (spy_results['dates'] >= start) & (spy_results['dates'] <= end)
        if mask.sum() == 0:
            continue

        # SPY
        spy_bh_crisis = spy_results['buyhold_returns'][mask]
        spy_ov_crisis = spy_results['portfolio_returns'][mask]
        spy_bh_ret = (1 + spy_bh_crisis).prod() - 1
        spy_ov_ret = (1 + spy_ov_crisis).prod() - 1

        # UPRO
        upro_bh_crisis = upro_results['buyhold_returns'][mask]
        upro_ov_crisis = upro_results['portfolio_returns'][mask]
        upro_bh_ret = (1 + upro_bh_crisis).prod() - 1
        upro_ov_ret = (1 + upro_ov_crisis).prod() - 1

        # Regime during crisis
        crisis_regime = spy_results['regime'][mask]
        regime_breakdown = crisis_regime.value_counts(normalize=True)

        print(f"\n  {crisis_name} ({start} to {end}):")
        print(f"    SPY  B&H: {spy_bh_ret:+.1%}  | Overlay: {spy_ov_ret:+.1%}  | Saved: {spy_ov_ret - spy_bh_ret:+.1%}")
        print(f"    UPRO B&H: {upro_bh_ret:+.1%}  | Overlay: {upro_ov_ret:+.1%}  | Saved: {upro_ov_ret - upro_bh_ret:+.1%}")
        regime_str = ', '.join([f"{r}: {v:.0%}" for r, v in regime_breakdown.items()])
        print(f"    Regime: {regime_str}")

    # ── Lookahead Validation ──
    m_t0, m_t1 = lookahead_comparison(prices)

    # ── Plots ──
    print("\nGenerating plots...")
    plot_equity_curves([spy_results, upro_results])
    plot_regime_timeline(signals)
    plot_drawdowns([spy_results, upro_results])

    # ── Save Results ──
    summary = {
        'backtest_period': f"{BACKTEST_START} to {END_DATE}",
        'signals_used': ['VIX_level', 'VIX_term_structure', 'credit_spread', 'market_breadth', 'SPY_trend'],
        'signal_lag': 'T-1 (honest)',
        'transaction_cost_bps': COST_BPS,
        'regime_distribution': {r: f"{regime_dist.get(r, 0):.1%}" for r in ['risk_on', 'neutral', 'risk_off']},
        'metrics': {
            'SPY_buyhold': spy_bh_metrics,
            'SPY_overlay': spy_ov_metrics,
            'UPRO_buyhold': upro_bh_metrics,
            'UPRO_overlay': upro_ov_metrics,
        },
        'lookahead_validation': {
            'T0_sharpe': m_t0['sharpe'],
            'T1_sharpe': m_t1['sharpe'],
            'sharpe_gap': round(m_t0['sharpe'] - m_t1['sharpe'], 2),
        },
        'crisis_protection': {},
    }

    # Re-compute crisis returns for JSON
    for crisis_name, start, end in crisis_periods:
        mask = (spy_results['dates'] >= start) & (spy_results['dates'] <= end)
        if mask.sum() == 0:
            continue
        spy_bh_ret = float((1 + spy_results['buyhold_returns'][mask]).prod() - 1)
        spy_ov_ret = float((1 + spy_results['portfolio_returns'][mask]).prod() - 1)
        upro_bh_ret = float((1 + upro_results['buyhold_returns'][mask]).prod() - 1)
        upro_ov_ret = float((1 + upro_results['portfolio_returns'][mask]).prod() - 1)
        summary['crisis_protection'][crisis_name] = {
            'SPY_bh': f"{spy_bh_ret:+.1%}",
            'SPY_overlay': f"{spy_ov_ret:+.1%}",
            'UPRO_bh': f"{upro_bh_ret:+.1%}",
            'UPRO_overlay': f"{upro_ov_ret:+.1%}",
        }

    with open(os.path.join(OUTPUT_DIR, 'backtest_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save signals for further analysis
    signals_export = signals[signals.index >= BACKTEST_START].copy()
    signals_export.to_parquet(os.path.join(OUTPUT_DIR, 'regime_signals.parquet'))

    print(f"\nAll results saved to {OUTPUT_DIR}/")
    print("\nDONE.")


if __name__ == '__main__':
    main()
