#!/usr/bin/env python3
"""
Adaptive Leveraged Growth with Drawdown Control v1
===================================================
HC #724: T-1 signals, T+1 execution. Anti-lookahead.
HC #718: Permutation tests, transaction costs, min 100 OOS days.

Core idea: Don't find new alpha — optimize what works.
UPRO when safe, QQQ when moderate, SHY when dangerous.
Multi-signal regime scoring with drawdown circuit breaker.

Regime signals (ALL T-1):
  1. SPY vs 200 SMA (trend)
  2. SPY vs 50 SMA (short-term trend)
  3. VIX level (<20, 20-30, >30)
  4. VIX term structure (VIX/VIX3M contango vs backwardation)
  5. Credit spread (HYG vs SHY 20d return differential)
  6. Sector breadth (% of sectors above 50d SMA)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import json
import os
import sys
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Allocation rules by regime score (0-6)
    'alloc_rules': {
        (5, 6): {'UPRO': 1.0},                    # All clear
        (3, 4): {'QQQ': 0.70, 'SHY': 0.30},       # Mostly clear
        (1, 2): {'SHY': 1.0},                      # Caution
        (0, 0): {'SHY': 1.0},                      # Crisis
    },
    # Transaction costs (bps per switch)
    'cost_upro_bps': 20,
    'cost_qqq_bps': 10,
    'cost_shy_bps': 10,
    # Rebalance throttle
    'max_rebal_per_week': 1,
    # Drawdown overlay (rolling 60d window, not all-time HWM)
    'dd_circuit_breaker': 0.15,   # Force SHY if rolling DD > 15%
    'dd_lookback': 60,            # Rolling window for DD calculation (trading days)
    'dd_cooldown_days': 10,       # Min days in SHY before re-entry
    'dd_reentry_min_score': 4,    # Min regime score to re-enter
    # SMA lookbacks
    'sma_200': 200,
    'sma_50': 50,
    'sector_sma': 50,
    # VIX thresholds
    'vix_low': 20,
    'vix_high': 30,
    # Breadth threshold
    'breadth_bullish': 0.6,  # 60% of sectors above SMA = bullish
    # Walk-forward
    'lookback_days': 252,
    # Permutation test
    'n_perms': 100,
    # Date range
    'start_date': '2009-01-01',  # Extra buffer for SMA calc
    'eval_start': '2010-01-04',  # Actual evaluation start
    'end_date': '2026-07-18',
}

SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data():
    """Download all required data from yfinance."""
    tickers = ['SPY', 'UPRO', 'QQQ', 'SHY', '^VIX', '^VIX3M', 'HYG'] + SECTOR_ETFS

    print(f"Downloading {len(tickers)} tickers from yfinance...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=CONFIG['start_date'], end=CONFIG['end_date'],
                           progress=False, auto_adjust=True)
            if len(df) > 0:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df['Close']
                print(f"  {t}: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
            else:
                print(f"  {t}: NO DATA")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")

    # Build combined DataFrame
    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    # Remove timezone if present
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()

    # Forward-fill small gaps (weekends/holidays already excluded)
    prices = prices.ffill(limit=5)

    print(f"\nCombined dataset: {len(prices)} rows, {prices.shape[1]} columns")
    print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
    print(f"Missing data:\n{prices.isnull().sum()}")

    return prices


# ============================================================
# REGIME SCORING (ALL T-1)
# ============================================================
def compute_regime_signals(prices):
    """
    Compute 6 regime signals. ALL use T-1 data only.
    Returns a DataFrame of binary signals (1=bullish, 0=bearish) and total score.
    """
    spy = prices['SPY']
    vix = prices['^VIX']

    signals = pd.DataFrame(index=prices.index)

    # 1. SPY > 200 SMA (trend)
    sma200 = spy.rolling(CONFIG['sma_200'], min_periods=CONFIG['sma_200']).mean()
    signals['trend_200'] = (spy > sma200).astype(int)

    # 2. SPY > 50 SMA (short-term trend)
    sma50 = spy.rolling(CONFIG['sma_50'], min_periods=CONFIG['sma_50']).mean()
    signals['trend_50'] = (spy > sma50).astype(int)

    # 3. VIX level
    signals['vix_calm'] = (vix < CONFIG['vix_low']).astype(int)

    # 4. VIX term structure (contango = bullish: VIX < VIX3M)
    if '^VIX3M' in prices.columns and prices['^VIX3M'].notna().sum() > 100:
        vix3m = prices['^VIX3M']
        signals['vix_contango'] = (vix < vix3m).astype(int)
    else:
        # Fallback: VIX below 20d SMA
        vix_sma = vix.rolling(20).mean()
        signals['vix_contango'] = (vix < vix_sma).astype(int)
        print("  WARNING: VIX3M unavailable, using VIX vs 20d SMA as proxy")

    # 5. Credit spread (HYG vs SHY 20d return diff — positive = credit improving)
    if 'HYG' in prices.columns and 'SHY' in prices.columns:
        hyg_ret20 = prices['HYG'].pct_change(20)
        shy_ret20 = prices['SHY'].pct_change(20)
        credit_diff = hyg_ret20 - shy_ret20
        signals['credit_ok'] = (credit_diff > 0).astype(int)
    else:
        signals['credit_ok'] = 1  # Default bullish if no data
        print("  WARNING: HYG/SHY unavailable for credit spread")

    # 6. Sector breadth (% of sectors above 50d SMA)
    available_sectors = [s for s in SECTOR_ETFS if s in prices.columns]
    if len(available_sectors) >= 5:
        breadth_scores = pd.DataFrame(index=prices.index)
        for s in available_sectors:
            sec_sma = prices[s].rolling(CONFIG['sector_sma'], min_periods=CONFIG['sector_sma']).mean()
            breadth_scores[s] = (prices[s] > sec_sma).astype(int)
        breadth_pct = breadth_scores.mean(axis=1)
        signals['breadth_ok'] = (breadth_pct >= CONFIG['breadth_bullish']).astype(int)
    else:
        signals['breadth_ok'] = 1
        print(f"  WARNING: Only {len(available_sectors)} sectors available, breadth signal defaulting")

    # CRITICAL: Shift ALL signals by 1 day (T-1 data for T+1 execution)
    signals = signals.shift(1)

    # Total regime score (0-6)
    signals['regime_score'] = signals[['trend_200', 'trend_50', 'vix_calm',
                                        'vix_contango', 'credit_ok', 'breadth_ok']].sum(axis=1)

    return signals


# ============================================================
# ALLOCATION ENGINE
# ============================================================
def get_target_allocation(score):
    """Map regime score to target allocation dict."""
    if score >= 5:
        return {'UPRO': 1.0}
    elif score >= 3:
        return {'QQQ': 0.70, 'SHY': 0.30}
    elif score >= 1:
        return {'SHY': 1.0}
    else:
        return {'SHY': 1.0}


def compute_transaction_cost(old_alloc, new_alloc):
    """Compute transaction cost in bps for rebalancing."""
    cost_map = {
        'UPRO': CONFIG['cost_upro_bps'],
        'QQQ': CONFIG['cost_qqq_bps'],
        'SHY': CONFIG['cost_shy_bps'],
    }

    all_assets = set(list(old_alloc.keys()) + list(new_alloc.keys()))
    total_turnover = 0
    weighted_cost = 0

    for asset in all_assets:
        old_w = old_alloc.get(asset, 0)
        new_w = new_alloc.get(asset, 0)
        delta = abs(new_w - old_w)
        total_turnover += delta
        weighted_cost += delta * cost_map.get(asset, 10)

    return weighted_cost / 10000  # Convert bps to decimal


def run_backtest(prices, signals, use_lag=True, shuffle_seed=None):
    """
    Run the adaptive leveraged growth backtest.

    Args:
        prices: Price DataFrame
        signals: Regime signal DataFrame
        use_lag: If True, use T-1 signals (proper). If False, use T-0 (lookahead test).
        shuffle_seed: If not None, shuffle regime scores for permutation test.

    Returns:
        DataFrame with daily returns and metadata.
    """
    eval_start = pd.Timestamp(CONFIG['eval_start'])

    # Get returns for tradeable assets
    asset_returns = pd.DataFrame(index=prices.index)
    for asset in ['UPRO', 'QQQ', 'SHY', 'SPY']:
        if asset in prices.columns:
            asset_returns[asset] = prices[asset].pct_change()

    # Regime scores
    if use_lag:
        scores = signals['regime_score'].copy()
    else:
        # T-0 test: unshift (use same-day signal — lookahead!)
        scores = signals['regime_score'].shift(-1)

    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        valid_idx = scores.dropna().index
        shuffled_vals = scores.loc[valid_idx].values.copy()
        rng.shuffle(shuffled_vals)
        scores.loc[valid_idx] = shuffled_vals

    # Simulation
    eval_mask = prices.index >= eval_start
    eval_dates = prices.index[eval_mask]

    equity = 1.0
    current_alloc = {'SHY': 1.0}
    circuit_breaker_active = False
    cb_trigger_day = 0  # Day index when circuit breaker triggered
    last_rebal_date = None
    equity_history = []  # Track for rolling DD

    results = []
    day_idx = 0

    for date in eval_dates:
        if date not in asset_returns.index:
            continue

        score = scores.get(date, np.nan)
        if np.isnan(score):
            # No signal — hold current allocation
            day_ret = sum(current_alloc.get(a, 0) * asset_returns.loc[date].get(a, 0)
                         for a in current_alloc)
            equity *= (1 + day_ret)
            equity_history.append(equity)
            results.append({
                'date': date, 'return': day_ret, 'equity': equity,
                'score': np.nan, 'alloc': str(current_alloc),
                'circuit_breaker': circuit_breaker_active, 'cost': 0,
            })
            day_idx += 1
            continue

        score = int(score)

        # Rolling drawdown: DD from max equity over last N trading days
        lookback = CONFIG['dd_lookback']
        recent_eq = equity_history[-lookback:] if len(equity_history) >= lookback else equity_history
        if len(recent_eq) > 0:
            rolling_hwm = max(recent_eq)
            rolling_dd = (rolling_hwm - equity) / rolling_hwm if rolling_hwm > 0 else 0
        else:
            rolling_dd = 0

        # Circuit breaker logic
        if not circuit_breaker_active:
            if rolling_dd > CONFIG['dd_circuit_breaker']:
                circuit_breaker_active = True
                cb_trigger_day = day_idx
        else:
            # Re-enter after cooldown period AND regime is supportive
            days_in_cb = day_idx - cb_trigger_day
            if days_in_cb >= CONFIG['dd_cooldown_days'] and score >= CONFIG['dd_reentry_min_score']:
                circuit_breaker_active = False

        # Determine target allocation
        if circuit_breaker_active:
            target_alloc = {'SHY': 1.0}
        else:
            target_alloc = get_target_allocation(score)

        # Rebalance throttle: max 1x per week
        needs_rebal = (target_alloc != current_alloc)
        can_rebal = True
        if needs_rebal and last_rebal_date is not None:
            days_since = (date - last_rebal_date).days
            if days_since < 5:  # ~1 week in trading days
                can_rebal = False

        # Apply rebalance with costs
        cost = 0
        if needs_rebal and can_rebal:
            cost = compute_transaction_cost(current_alloc, target_alloc)
            current_alloc = target_alloc.copy()
            last_rebal_date = date

        # Compute daily return
        day_ret = sum(current_alloc.get(a, 0) * asset_returns.loc[date].get(a, 0)
                     for a in current_alloc)
        day_ret -= cost  # Subtract transaction cost

        equity *= (1 + day_ret)
        equity_history.append(equity)

        # Track all-time HWM for drawdown reporting
        all_time_hwm = max(equity_history)
        all_time_dd = (all_time_hwm - equity) / all_time_hwm if all_time_hwm > 0 else 0

        day_idx += 1

        results.append({
            'date': date, 'return': day_ret, 'equity': equity,
            'score': score, 'alloc': str(current_alloc),
            'circuit_breaker': circuit_breaker_active, 'cost': cost,
            'drawdown': all_time_dd, 'rolling_dd': rolling_dd,
        })

    return pd.DataFrame(results).set_index('date')


# ============================================================
# BENCHMARK STRATEGIES
# ============================================================
def run_benchmarks(prices):
    """Run benchmark strategies for comparison."""
    eval_start = pd.Timestamp(CONFIG['eval_start'])

    benchmarks = {}

    for asset in ['SPY', 'UPRO']:
        if asset in prices.columns:
            rets = prices[asset].pct_change()
            mask = prices.index >= eval_start
            eq = (1 + rets[mask]).cumprod()
            benchmarks[f'{asset} B&H'] = rets[mask]

    # 60/40 (SPY/SHY rebalanced monthly)
    if 'SPY' in prices.columns and 'SHY' in prices.columns:
        spy_ret = prices['SPY'].pct_change()
        shy_ret = prices['SHY'].pct_change()
        mask = prices.index >= eval_start
        blend = 0.6 * spy_ret[mask] + 0.4 * shy_ret[mask]
        benchmarks['60/40'] = blend

    # Simple UPRO + 200 SMA
    if 'UPRO' in prices.columns and 'SPY' in prices.columns:
        spy = prices['SPY']
        sma200 = spy.rolling(200).mean()
        signal = (spy > sma200).shift(1).fillna(0).astype(int)  # T-1
        upro_ret = prices['UPRO'].pct_change()
        shy_ret = prices['SHY'].pct_change() if 'SHY' in prices.columns else 0
        mask = prices.index >= eval_start
        strat_ret = signal[mask] * upro_ret[mask] + (1 - signal[mask]) * shy_ret[mask]
        benchmarks['UPRO+200SMA'] = strat_ret

    return benchmarks


# ============================================================
# ANALYTICS
# ============================================================
def compute_metrics(returns, name='Strategy', ann_factor=252):
    """Compute standard performance metrics."""
    returns = returns.dropna()
    if len(returns) == 0:
        return {'name': name, 'error': 'no data'}

    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / ann_factor
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    vol = returns.std() * np.sqrt(ann_factor)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor) if returns.std() > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(ann_factor) if (returns < 0).any() else 0.001
    sortino = (returns.mean() * ann_factor) / downside

    # Max drawdown
    eq = (1 + returns).cumprod()
    hwm = eq.cummax()
    dd = (eq - hwm) / hwm
    max_dd = dd.min()

    # Win rate
    wr = (returns > 0).mean()

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else np.inf

    # Calmar ratio
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    return {
        'name': name,
        'total_return': total_ret,
        'cagr': cagr,
        'volatility': vol,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_drawdown': max_dd,
        'calmar': calmar,
        'win_rate': wr,
        'profit_factor': pf,
        'n_days': len(returns),
        'n_years': n_years,
    }


def year_by_year(results_df, benchmarks):
    """Year-by-year performance breakdown."""
    print("\n" + "="*100)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("="*100)

    strategy_rets = results_df['return']

    years = sorted(strategy_rets.index.year.unique())

    header = f"{'Year':<6} {'Strategy':>12} {'Sharpe':>8} {'MaxDD':>8} {'WR':>6}"
    for bname in list(benchmarks.keys())[:3]:
        header += f" | {bname:>12}"
    print(header)
    print("-" * len(header))

    for year in years:
        mask = strategy_rets.index.year == year
        yr_ret = strategy_rets[mask]
        if len(yr_ret) < 10:
            continue

        m = compute_metrics(yr_ret, f'{year}')
        line = f"{year:<6} {m['total_return']:>11.1%} {m['sharpe']:>8.2f} {m['max_drawdown']:>8.1%} {m['win_rate']:>5.1%}"

        for bname, brets in list(benchmarks.items())[:3]:
            bmask = brets.index.year == year
            br = brets[bmask]
            if len(br) > 0:
                bm = compute_metrics(br, bname)
                line += f" | {bm['total_return']:>11.1%}"
            else:
                line += f" | {'N/A':>12}"
        print(line)

    # Highlight key years
    print(f"\n  Key years: 2020 (COVID), 2022 (bear), 2024 (recovery)")


def regime_analysis(results_df, prices):
    """Analyze performance by regime and SPY market direction."""
    print("\n" + "="*80)
    print("REGIME-AGNOSTIC VALIDATION")
    print("="*80)

    spy_ret = prices['SPY'].pct_change()

    # Classify days by SPY performance
    eval_start = pd.Timestamp(CONFIG['eval_start'])
    spy_daily = spy_ret[spy_ret.index >= eval_start]

    # Green/Red/Flat classification
    green_days = spy_daily[spy_daily > 0.001].index
    red_days = spy_daily[spy_daily < -0.001].index
    flat_days = spy_daily[(spy_daily >= -0.001) & (spy_daily <= 0.001)].index

    strat_ret = results_df['return']

    for label, days in [('GREEN (SPY > +10bps)', green_days),
                         ('RED (SPY < -10bps)', red_days),
                         ('FLAT', flat_days)]:
        mask = strat_ret.index.isin(days)
        if mask.sum() > 10:
            m = compute_metrics(strat_ret[mask], label)
            print(f"  {label:30s}: Sharpe={m['sharpe']:+.2f}  Sortino={m['sortino']:+.2f}  "
                  f"WR={m['win_rate']:.1%}  N={m['n_days']}")

    # Regime balance check (HC #428 R1)
    green_m = compute_metrics(strat_ret[strat_ret.index.isin(green_days)], 'green')
    red_m = compute_metrics(strat_ret[strat_ret.index.isin(red_days)], 'red')

    if green_m.get('sharpe', 0) and red_m.get('sharpe', 0):
        s_green = green_m['sharpe']
        s_red = red_m['sharpe']
        imbalance = abs(s_green - s_red) / max(abs(s_green), abs(s_red), 0.001)
        status = "PASS" if imbalance <= 0.50 else "FAIL"
        print(f"\n  Regime imbalance: |{s_green:.2f} - {s_red:.2f}| / max = {imbalance:.2f} [{status}]")
        print(f"  (HC #428 R1: must be <= 0.50)")


def worst_drawdown_episodes(results_df, top_n=5):
    """Detail the worst drawdown episodes."""
    print("\n" + "="*80)
    print(f"TOP {top_n} WORST DRAWDOWN EPISODES")
    print("="*80)

    eq = results_df['equity']
    hwm = eq.cummax()
    dd = (eq - hwm) / hwm

    # Find drawdown episodes
    in_dd = dd < 0
    episodes = []
    start = None

    for i, (date, val) in enumerate(dd.items()):
        if val < 0 and start is None:
            start = date
        elif val >= 0 and start is not None:
            # End of episode
            episode_dd = dd[start:date]
            worst = episode_dd.min()
            worst_date = episode_dd.idxmin()
            duration = (date - start).days
            episodes.append({
                'start': start, 'end': date, 'trough': worst_date,
                'max_dd': worst, 'duration_days': duration,
            })
            start = None

    # Handle ongoing drawdown
    if start is not None:
        episode_dd = dd[start:]
        worst = episode_dd.min()
        worst_date = episode_dd.idxmin()
        duration = (dd.index[-1] - start).days
        episodes.append({
            'start': start, 'end': dd.index[-1], 'trough': worst_date,
            'max_dd': worst, 'duration_days': duration,
        })

    episodes.sort(key=lambda x: x['max_dd'])

    for i, ep in enumerate(episodes[:top_n]):
        print(f"  #{i+1}: {ep['max_dd']:.1%} drawdown")
        print(f"       Start: {ep['start'].strftime('%Y-%m-%d')}, "
              f"Trough: {ep['trough'].strftime('%Y-%m-%d')}, "
              f"End: {ep['end'].strftime('%Y-%m-%d')}")
        print(f"       Duration: {ep['duration_days']} calendar days")


def monthly_return_distribution(results_df):
    """Monthly return distribution analysis."""
    print("\n" + "="*80)
    print("MONTHLY RETURN DISTRIBUTION")
    print("="*80)

    monthly = results_df['return'].resample('ME').apply(lambda x: (1+x).prod()-1)

    print(f"  N months: {len(monthly)}")
    print(f"  Mean:     {monthly.mean():.2%}")
    print(f"  Median:   {monthly.median():.2%}")
    print(f"  Std:      {monthly.std():.2%}")
    print(f"  Skew:     {monthly.skew():.2f}")
    print(f"  Kurt:     {monthly.kurtosis():.2f}")
    print(f"  Best:     {monthly.max():.2%} ({monthly.idxmax().strftime('%Y-%m')})")
    print(f"  Worst:    {monthly.min():.2%} ({monthly.idxmin().strftime('%Y-%m')})")
    print(f"  % positive months: {(monthly > 0).mean():.1%}")

    # Histogram buckets
    bins = [-1, -0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10, 1]
    labels = ['<-10%', '-10/-5%', '-5/-2%', '-2/0%', '0/+2%', '+2/+5%', '+5/+10%', '>+10%']
    hist = pd.cut(monthly, bins=bins, labels=labels).value_counts().sort_index()
    print(f"\n  Distribution:")
    for bucket, count in hist.items():
        bar = '#' * count
        print(f"    {bucket:>10s}: {count:>3d} {bar}")


def lag_sensitivity_test(prices, signals):
    """Test T-0 vs T-1 signals to detect lookahead bias."""
    print("\n" + "="*80)
    print("LAG SENSITIVITY TEST (HC #724)")
    print("="*80)

    # T-1 (proper)
    res_t1 = run_backtest(prices, signals, use_lag=True)
    m_t1 = compute_metrics(res_t1['return'], 'T-1 (proper)')

    # T-0 (lookahead — should NOT be massively better)
    res_t0 = run_backtest(prices, signals, use_lag=False)
    m_t0 = compute_metrics(res_t0['return'], 'T-0 (lookahead)')

    print(f"  T-1 (no lookahead): Sharpe={m_t1['sharpe']:.3f}  CAGR={m_t1['cagr']:.1%}  MaxDD={m_t1['max_drawdown']:.1%}")
    print(f"  T-0 (lookahead):    Sharpe={m_t0['sharpe']:.3f}  CAGR={m_t0['cagr']:.1%}  MaxDD={m_t0['max_drawdown']:.1%}")

    sharpe_ratio = m_t0['sharpe'] / max(m_t1['sharpe'], 0.001)
    print(f"\n  T-0/T-1 Sharpe ratio: {sharpe_ratio:.2f}")
    if sharpe_ratio > 2.0:
        print(f"  WARNING: T-0 is >2x better — possible lookahead bias in signals!")
    else:
        print(f"  OK: T-0 advantage is modest — signals are genuinely predictive at T-1.")

    return m_t1, m_t0


def permutation_test(prices, signals, n_perms=100):
    """
    Shuffle regime scores across dates to test if edge is real.
    HC #718: 100 permutations minimum.
    """
    print("\n" + "="*80)
    print(f"PERMUTATION TEST ({n_perms} shuffles) — HC #718")
    print("="*80)

    # Real strategy Sharpe
    real_res = run_backtest(prices, signals, use_lag=True)
    real_sharpe = compute_metrics(real_res['return'], 'Real')['sharpe']

    # Shuffled Sharpes
    perm_sharpes = []
    for i in range(n_perms):
        if (i + 1) % 20 == 0:
            print(f"  Permutation {i+1}/{n_perms}...")
        perm_res = run_backtest(prices, signals, use_lag=True, shuffle_seed=i)
        perm_m = compute_metrics(perm_res['return'], f'perm_{i}')
        perm_sharpes.append(perm_m['sharpe'])

    perm_sharpes = np.array(perm_sharpes)

    # p-value: fraction of permutations >= real
    p_value = (perm_sharpes >= real_sharpe).mean()

    print(f"\n  Real strategy Sharpe:      {real_sharpe:.3f}")
    print(f"  Permuted mean Sharpe:      {perm_sharpes.mean():.3f}")
    print(f"  Permuted median Sharpe:    {np.median(perm_sharpes):.3f}")
    print(f"  Permuted std Sharpe:       {perm_sharpes.std():.3f}")
    print(f"  Permuted 95th percentile:  {np.percentile(perm_sharpes, 95):.3f}")
    print(f"  Permuted max Sharpe:       {perm_sharpes.max():.3f}")
    print(f"\n  p-value: {p_value:.3f}")

    if p_value < 0.05:
        print(f"  RESULT: SIGNIFICANT (p < 0.05) — signals contain real information")
    elif p_value < 0.10:
        print(f"  RESULT: MARGINAL (0.05 < p < 0.10) — weak evidence")
    else:
        print(f"  RESULT: NOT SIGNIFICANT (p >= 0.10) — no evidence signals help")

    return real_sharpe, perm_sharpes, p_value


def signal_contribution_analysis(prices, signals):
    """Test each signal's individual contribution."""
    print("\n" + "="*80)
    print("INDIVIDUAL SIGNAL CONTRIBUTION (leave-one-out)")
    print("="*80)

    signal_cols = ['trend_200', 'trend_50', 'vix_calm', 'vix_contango', 'credit_ok', 'breadth_ok']

    # Full model Sharpe
    full_res = run_backtest(prices, signals, use_lag=True)
    full_sharpe = compute_metrics(full_res['return'], 'Full')['sharpe']
    print(f"  Full model (6 signals): Sharpe = {full_sharpe:.3f}")
    print()

    for col in signal_cols:
        # Create modified signals with one signal zeroed out
        mod_signals = signals.copy()
        mod_signals[col] = 0
        mod_signals['regime_score'] = mod_signals[signal_cols].sum(axis=1)
        # Re-apply the T-1 shift is already done, but regime_score needs recompute
        # Actually signals already have the shift applied, so just recompute score

        mod_res = run_backtest(prices, mod_signals, use_lag=True)
        mod_sharpe = compute_metrics(mod_res['return'], f'w/o {col}')['sharpe']
        delta = full_sharpe - mod_sharpe
        direction = "+" if delta > 0 else "-"
        print(f"  Without {col:>15s}: Sharpe = {mod_sharpe:.3f}  (delta = {direction}{abs(delta):.3f})")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 80)
    print("ADAPTIVE LEVERAGED GROWTH WITH DRAWDOWN CONTROL v1")
    print("=" * 80)
    print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Eval period: {CONFIG['eval_start']} to {CONFIG['end_date']}")
    print()

    # 1. Download data
    prices = download_data()

    # 2. Compute regime signals
    print("\nComputing regime signals (all T-1)...")
    signals = compute_regime_signals(prices)

    signal_cols = ['trend_200', 'trend_50', 'vix_calm', 'vix_contango', 'credit_ok', 'breadth_ok']
    eval_mask = prices.index >= pd.Timestamp(CONFIG['eval_start'])
    for col in signal_cols:
        pct_bullish = signals.loc[eval_mask, col].mean()
        print(f"  {col:>15s}: {pct_bullish:.1%} bullish")

    score_dist = signals.loc[eval_mask, 'regime_score'].value_counts().sort_index()
    print(f"\n  Regime score distribution:")
    for score, count in score_dist.items():
        pct = count / eval_mask.sum()
        print(f"    Score {score:.0f}: {count:>5d} days ({pct:.1%})")

    # 3. Run main backtest
    print("\n" + "="*80)
    print("MAIN BACKTEST")
    print("="*80)

    results = run_backtest(prices, signals, use_lag=True)
    main_metrics = compute_metrics(results['return'], 'Adaptive Leveraged Growth')

    print(f"\n  === STRATEGY: Adaptive Leveraged Growth ===")
    print(f"  CAGR:           {main_metrics['cagr']:.1%}")
    print(f"  Volatility:     {main_metrics['volatility']:.1%}")
    print(f"  Sharpe:         {main_metrics['sharpe']:.3f}")
    print(f"  Sortino:        {main_metrics['sortino']:.3f}")
    print(f"  Max Drawdown:   {main_metrics['max_drawdown']:.1%}")
    print(f"  Calmar:         {main_metrics['calmar']:.3f}")
    print(f"  Win Rate:       {main_metrics['win_rate']:.1%}")
    print(f"  Profit Factor:  {main_metrics['profit_factor']:.3f}")
    print(f"  Total Return:   {main_metrics['total_return']:.1%}")
    print(f"  N days:         {main_metrics['n_days']}")

    # Transaction cost impact
    total_costs = results['cost'].sum()
    n_rebalances = (results['cost'] > 0).sum()
    print(f"\n  Total transaction costs: {total_costs:.4f} ({total_costs*100:.2f}% drag)")
    print(f"  Number of rebalances: {n_rebalances}")
    print(f"  Circuit breaker activations: {results['circuit_breaker'].sum()} days")

    # 4. Benchmarks
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)

    benchmarks = run_benchmarks(prices)

    all_metrics = [main_metrics]
    for bname, brets in benchmarks.items():
        bm = compute_metrics(brets, bname)
        all_metrics.append(bm)

    print(f"\n  {'Strategy':<25s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'WR':>6s}")
    print("  " + "-" * 73)
    for m in all_metrics:
        if 'error' in m:
            continue
        print(f"  {m['name']:<25s} {m['cagr']:>7.1%} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['max_drawdown']:>8.1%} {m['calmar']:>8.3f} {m['win_rate']:>5.1%}")

    # 5. Year-by-year
    year_by_year(results, benchmarks)

    # 6. Regime analysis (HC #428 R1)
    regime_analysis(results, prices)

    # 7. Lag sensitivity (HC #724)
    m_t1, m_t0 = lag_sensitivity_test(prices, signals)

    # 8. Permutation test (HC #718)
    real_sharpe, perm_sharpes, p_value = permutation_test(prices, signals, n_perms=CONFIG['n_perms'])

    # 9. Signal contribution
    signal_contribution_analysis(prices, signals)

    # 10. Monthly returns
    monthly_return_distribution(results)

    # 11. Worst drawdowns
    worst_drawdown_episodes(results)

    # ============================================================
    # SUMMARY VERDICT
    # ============================================================
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    checks = {
        'Sharpe > 0.5': main_metrics['sharpe'] > 0.5,
        'Beats SPY B&H Sharpe': main_metrics['sharpe'] > compute_metrics(benchmarks.get('SPY B&H', pd.Series()), 'SPY').get('sharpe', 0),
        'MaxDD < -50%': main_metrics['max_drawdown'] > -0.50,
        'Perm test p < 0.05': p_value < 0.05,
        'T-0/T-1 ratio < 2.0': (m_t0['sharpe'] / max(m_t1['sharpe'], 0.001)) < 2.0,
        'OOS days >= 100': main_metrics['n_days'] >= 100,
        'Win rate > 50%': main_metrics['win_rate'] > 0.50,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    n_pass = sum(checks.values())
    n_total = len(checks)
    print(f"\n  Score: {n_pass}/{n_total} checks passed")

    if n_pass == n_total:
        print("  VERDICT: ALL CHECKS PASSED — strategy is robust")
    elif n_pass >= 5:
        print("  VERDICT: MOSTLY PASSING — review failed checks")
    else:
        print("  VERDICT: SIGNIFICANT ISSUES — do not deploy")

    # Save results
    output = {
        'strategy': 'Adaptive Leveraged Growth v1',
        'run_time': datetime.now().isoformat(),
        'config': {k: str(v) for k, v in CONFIG.items()},
        'metrics': main_metrics,
        'benchmarks': {m['name']: m for m in all_metrics[1:]},
        'perm_test': {
            'real_sharpe': float(real_sharpe),
            'p_value': float(p_value),
            'perm_mean': float(perm_sharpes.mean()),
            'perm_std': float(perm_sharpes.std()),
        },
        'checks': {k: bool(v) for k, v in checks.items()},
        'checks_passed': n_pass,
        'checks_total': n_total,
    }

    # Convert any numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    output_path = RESULTS_DIR / 'adaptive_leveraged_growth_v1_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\n  Results saved to {output_path}")
    print(f"\n{'='*80}")
    print("DONE")
    print(f"{'='*80}")

    return output


if __name__ == '__main__':
    main()
