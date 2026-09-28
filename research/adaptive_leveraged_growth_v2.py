#!/usr/bin/env python3
"""
Adaptive Leveraged Growth v2 — VIX-Overlay UPRO
================================================
HC #724: T-1 signals, T+1 execution. Anti-lookahead.
HC #718: Permutation tests, transaction costs, min 100 OOS.

LESSON FROM v1: 6-signal regime scoring added complexity without edge.
Simple UPRO+200SMA (Sharpe 0.82) already works. 91.5% cost drag killed v1.

v2 APPROACH: Take what works and protect it.
- Base: UPRO when SPY > 200 SMA (T-1)
- VIX overlay: Step down leverage in high-vol regimes
- Minimal switching: Weekly rebalance only, no daily whipsaws
- Drawdown circuit breaker with rolling window

Allocation (all signals T-1):
- SPY > 200 SMA AND VIX < 20:       100% UPRO (full risk-on)
- SPY > 200 SMA AND 20 <= VIX < 30: 100% QQQ  (risk-on but deleveraged)
- SPY > 200 SMA AND VIX >= 30:      100% SHY  (trend up but panic — wait)
- SPY <= 200 SMA:                    100% SHY  (risk-off)

VIX term structure bonus:
- If VIX/VIX3M > 1.0 (backwardation = stress): downgrade one notch

This is 3 signals total (200SMA, VIX level, VIX term structure) = max 8 regime combos.
Minimizes transaction costs by only switching between 3 ETFs.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

CONFIG = {
    'sma_period': 200,
    'vix_low': 20,
    'vix_high': 30,
    'cost_upro_bps': 20,
    'cost_qqq_bps': 10,
    'cost_shy_bps': 5,
    'rebal_day': 'Friday',  # Only rebalance on Fridays (weekly)
    'dd_circuit_breaker': 0.20,  # 20% rolling DD triggers SHY
    'dd_lookback': 40,  # 40 trading days for rolling DD
    'dd_cooldown_days': 15,  # 15 trading days before re-entry
    'start_date': '2009-01-01',
    'eval_start': '2010-01-04',
    'end_date': '2026-07-18',
    'n_perms': 100,
}

RESULTS_DIR = Path('/home/jupiter/Lvl3Quant/research/findings')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def download_data():
    tickers = ['SPY', 'UPRO', 'QQQ', 'SHY', '^VIX', '^VIX3M']
    print(f"Downloading {len(tickers)} tickers...")
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=CONFIG['start_date'], end=CONFIG['end_date'],
                           progress=False, auto_adjust=True)
            if len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[t] = df['Close']
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")

    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index().ffill(limit=5)
    return prices


def compute_signals(prices):
    """3 signals, all T-1 shifted."""
    spy = prices['SPY']
    vix = prices['^VIX']

    signals = pd.DataFrame(index=prices.index)

    # Signal 1: SPY > 200 SMA
    sma200 = spy.rolling(CONFIG['sma_period'], min_periods=CONFIG['sma_period']).mean()
    signals['above_sma'] = (spy > sma200).astype(float)

    # Signal 2: VIX level category (0=high, 1=medium, 2=low)
    signals['vix_level'] = 1  # medium default
    signals.loc[vix < CONFIG['vix_low'], 'vix_level'] = 2  # low VIX = bullish
    signals.loc[vix >= CONFIG['vix_high'], 'vix_level'] = 0  # high VIX = bearish

    # Signal 3: VIX term structure (contango = bullish)
    if '^VIX3M' in prices.columns and prices['^VIX3M'].notna().sum() > 100:
        vix3m = prices['^VIX3M']
        signals['vix_contango'] = (vix < vix3m).astype(float)  # 1 if contango
    else:
        signals['vix_contango'] = 1.0
        print("  WARNING: VIX3M unavailable")

    # CRITICAL: T-1 shift
    signals = signals.shift(1)

    return signals


def get_allocation(above_sma, vix_level, vix_contango):
    """Map signals to allocation."""
    if np.isnan(above_sma) or np.isnan(vix_level):
        return None  # No signal

    if above_sma < 0.5:
        # Below 200 SMA — risk off
        return 'SHY'

    # Above 200 SMA — check VIX
    if vix_level == 2:  # VIX < 20
        if vix_contango >= 0.5:
            return 'UPRO'  # All clear
        else:
            return 'QQQ'  # VIX low but backwardation = caution
    elif vix_level == 1:  # 20 <= VIX < 30
        return 'QQQ'  # Moderate vol
    else:  # VIX >= 30
        return 'SHY'  # Panic even if trend is up


def run_backtest(prices, signals, use_lag=True, shuffle_seed=None, weekly_only=True):
    """Run the strategy backtest."""
    eval_start = pd.Timestamp(CONFIG['eval_start'])

    asset_returns = pd.DataFrame(index=prices.index)
    for asset in ['UPRO', 'QQQ', 'SHY', 'SPY']:
        if asset in prices.columns:
            asset_returns[asset] = prices[asset].pct_change()

    sig = signals.copy()
    if not use_lag:
        # Undo the T-1 shift for lookahead test
        sig = sig.shift(-1)

    if shuffle_seed is not None:
        rng = np.random.RandomState(shuffle_seed)
        valid_idx = sig['above_sma'].dropna().index
        for col in ['above_sma', 'vix_level', 'vix_contango']:
            vals = sig.loc[valid_idx, col].values.copy()
            rng.shuffle(vals)
            sig.loc[valid_idx, col] = vals

    eval_mask = prices.index >= eval_start
    eval_dates = prices.index[eval_mask]

    equity = 1.0
    current_asset = 'SHY'
    equity_history = []
    circuit_breaker = False
    cb_trigger_day = 0
    day_idx = 0

    results = []

    for date in eval_dates:
        if date not in asset_returns.index:
            continue

        # Determine target allocation
        row = sig.loc[date] if date in sig.index else None
        if row is not None and not row.isna().all():
            target = get_allocation(row['above_sma'], row['vix_level'], row['vix_contango'])
        else:
            target = None

        # Rolling drawdown check
        lookback = CONFIG['dd_lookback']
        recent_eq = equity_history[-lookback:] if len(equity_history) >= lookback else equity_history
        if len(recent_eq) > 0:
            rolling_hwm = max(recent_eq)
            rolling_dd = (rolling_hwm - equity) / rolling_hwm if rolling_hwm > 0 else 0
        else:
            rolling_dd = 0

        # Circuit breaker
        if not circuit_breaker:
            if rolling_dd > CONFIG['dd_circuit_breaker']:
                circuit_breaker = True
                cb_trigger_day = day_idx
        else:
            if day_idx - cb_trigger_day >= CONFIG['dd_cooldown_days']:
                if target and target != 'SHY':
                    circuit_breaker = False

        if circuit_breaker:
            target = 'SHY'

        # Weekly rebalance throttle
        is_rebal_day = True
        if weekly_only:
            is_rebal_day = (date.day_name() == CONFIG['rebal_day'])

        cost = 0
        if target and target != current_asset and is_rebal_day:
            # Compute switching cost
            cost_bps = max(
                CONFIG.get(f'cost_{current_asset.lower()}_bps', 10),
                CONFIG.get(f'cost_{target.lower()}_bps', 10)
            )
            cost = cost_bps / 10000
            current_asset = target

        # Daily return
        day_ret = asset_returns.loc[date].get(current_asset, 0)
        if np.isnan(day_ret):
            day_ret = 0
        day_ret -= cost

        equity *= (1 + day_ret)
        equity_history.append(equity)
        day_idx += 1

        results.append({
            'date': date, 'return': day_ret, 'equity': equity,
            'asset': current_asset, 'circuit_breaker': circuit_breaker,
            'cost': cost, 'rolling_dd': rolling_dd,
        })

    return pd.DataFrame(results).set_index('date')


def run_benchmarks(prices):
    """Benchmark strategies."""
    eval_start = pd.Timestamp(CONFIG['eval_start'])
    benchmarks = {}

    for asset in ['SPY', 'UPRO']:
        if asset in prices.columns:
            rets = prices[asset].pct_change()
            mask = prices.index >= eval_start
            benchmarks[f'{asset} B&H'] = rets[mask]

    # 60/40
    if 'SPY' in prices.columns and 'SHY' in prices.columns:
        mask = prices.index >= eval_start
        benchmarks['60/40'] = (0.6 * prices['SPY'].pct_change() +
                               0.4 * prices['SHY'].pct_change())[mask]

    # Simple UPRO+200SMA
    if 'UPRO' in prices.columns:
        spy = prices['SPY']
        sma200 = spy.rolling(200).mean()
        sig = (spy > sma200).shift(1).fillna(0).astype(int)
        mask = prices.index >= eval_start
        upro_ret = prices['UPRO'].pct_change()
        shy_ret = prices['SHY'].pct_change()
        benchmarks['UPRO+200SMA'] = (sig * upro_ret + (1 - sig) * shy_ret)[mask]

    # QQQ B&H
    if 'QQQ' in prices.columns:
        mask = prices.index >= eval_start
        benchmarks['QQQ B&H'] = prices['QQQ'].pct_change()[mask]

    return benchmarks


def compute_metrics(returns, name='Strategy'):
    returns = returns.dropna()
    if len(returns) == 0:
        return {'name': name, 'sharpe': 0, 'cagr': 0, 'max_drawdown': 0,
                'sortino': 0, 'win_rate': 0, 'profit_factor': 0, 'calmar': 0,
                'total_return': 0, 'volatility': 0, 'n_days': 0, 'n_years': 0}

    total = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total) ** (1 / max(n_years, 0.01)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    downside = returns[returns < 0].std() * np.sqrt(252) if (returns < 0).any() else 0.001
    sortino = returns.mean() * 252 / downside
    eq = (1 + returns).cumprod()
    max_dd = (eq / eq.cummax() - 1).min()
    wr = (returns > 0).mean()
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else np.inf
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    return {
        'name': name, 'total_return': total, 'cagr': cagr, 'volatility': vol,
        'sharpe': sharpe, 'sortino': sortino, 'max_drawdown': max_dd,
        'calmar': calmar, 'win_rate': wr, 'profit_factor': pf,
        'n_days': len(returns), 'n_years': n_years,
    }


def print_comparison(all_metrics):
    print(f"\n  {'Strategy':<25s} {'CAGR':>8s} {'Sharpe':>8s} {'Sortino':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'WR':>6s}")
    print("  " + "-" * 73)
    for m in all_metrics:
        print(f"  {m['name']:<25s} {m['cagr']:>7.1%} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['max_drawdown']:>8.1%} {m['calmar']:>8.3f} {m['win_rate']:>5.1%}")


def year_by_year(results_df, benchmarks):
    print("\n" + "="*100)
    print("YEAR-BY-YEAR PERFORMANCE")
    print("="*100)

    rets = results_df['return']
    years = sorted(rets.index.year.unique())

    header = f"{'Year':<6} {'Strategy':>12} {'Sharpe':>8} {'MaxDD':>8} {'WR':>6}"
    for bname in list(benchmarks.keys())[:4]:
        header += f" | {bname:>12}"
    print(header)
    print("-" * len(header))

    for year in years:
        mask = rets.index.year == year
        yr = rets[mask]
        if len(yr) < 10:
            continue
        m = compute_metrics(yr, str(year))
        line = f"{year:<6} {m['total_return']:>11.1%} {m['sharpe']:>8.2f} {m['max_drawdown']:>8.1%} {m['win_rate']:>5.1%}"
        for bname, brets in list(benchmarks.items())[:4]:
            bmask = brets.index.year == year
            br = brets[bmask]
            if len(br) > 0:
                bm = compute_metrics(br, bname)
                line += f" | {bm['total_return']:>11.1%}"
            else:
                line += f" | {'N/A':>12}"
        print(line)


def regime_analysis(results_df, prices):
    print("\n" + "="*80)
    print("REGIME-AGNOSTIC VALIDATION (HC #428 R1)")
    print("="*80)

    eval_start = pd.Timestamp(CONFIG['eval_start'])
    spy_ret = prices['SPY'].pct_change()
    spy_daily = spy_ret[spy_ret.index >= eval_start]
    strat_ret = results_df['return']

    green = spy_daily[spy_daily > 0.001].index
    red = spy_daily[spy_daily < -0.001].index
    flat = spy_daily[(spy_daily >= -0.001) & (spy_daily <= 0.001)].index

    for label, days in [('GREEN', green), ('RED', red), ('FLAT', flat)]:
        mask = strat_ret.index.isin(days)
        if mask.sum() > 10:
            m = compute_metrics(strat_ret[mask], label)
            print(f"  {label:8s}: Sharpe={m['sharpe']:+.2f}  Sortino={m['sortino']:+.2f}  WR={m['win_rate']:.1%}  N={m['n_days']}")

    g = compute_metrics(strat_ret[strat_ret.index.isin(green)], 'g')
    r = compute_metrics(strat_ret[strat_ret.index.isin(red)], 'r')
    if g['sharpe'] and r['sharpe']:
        imb = abs(g['sharpe'] - r['sharpe']) / max(abs(g['sharpe']), abs(r['sharpe']), 0.001)
        print(f"\n  Regime imbalance: {imb:.2f} ({'PASS' if imb <= 0.50 else 'FAIL'}, threshold 0.50)")


def lag_sensitivity(prices, signals):
    print("\n" + "="*80)
    print("LAG SENSITIVITY TEST (HC #724)")
    print("="*80)

    res_t1 = run_backtest(prices, signals, use_lag=True)
    m_t1 = compute_metrics(res_t1['return'], 'T-1')

    res_t0 = run_backtest(prices, signals, use_lag=False)
    m_t0 = compute_metrics(res_t0['return'], 'T-0')

    print(f"  T-1 (proper):    Sharpe={m_t1['sharpe']:.3f}  CAGR={m_t1['cagr']:.1%}  MaxDD={m_t1['max_drawdown']:.1%}")
    print(f"  T-0 (lookahead): Sharpe={m_t0['sharpe']:.3f}  CAGR={m_t0['cagr']:.1%}  MaxDD={m_t0['max_drawdown']:.1%}")

    ratio = m_t0['sharpe'] / max(m_t1['sharpe'], 0.001)
    print(f"  Ratio: {ratio:.2f} ({'PASS' if ratio < 2.0 else 'FAIL'}, threshold 2.0)")
    return m_t1, m_t0, ratio


def permutation_test(prices, signals, n=100):
    print("\n" + "="*80)
    print(f"PERMUTATION TEST ({n} shuffles) — HC #718")
    print("="*80)

    real = run_backtest(prices, signals, use_lag=True)
    real_sharpe = compute_metrics(real['return'], 'Real')['sharpe']

    perm_sharpes = []
    for i in range(n):
        if (i+1) % 25 == 0:
            print(f"  {i+1}/{n}...")
        res = run_backtest(prices, signals, use_lag=True, shuffle_seed=i)
        perm_sharpes.append(compute_metrics(res['return'], f'p{i}')['sharpe'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    print(f"\n  Real Sharpe:   {real_sharpe:.3f}")
    print(f"  Perm mean:     {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}")
    print(f"  Perm 95th:     {np.percentile(perm_sharpes, 95):.3f}")
    print(f"  p-value:       {p_value:.3f}")
    print(f"  Result:        {'SIGNIFICANT' if p_value < 0.05 else 'MARGINAL' if p_value < 0.10 else 'NOT SIGNIFICANT'}")

    return real_sharpe, perm_sharpes, p_value


def allocation_stats(results_df):
    print("\n" + "="*80)
    print("ALLOCATION STATISTICS")
    print("="*80)

    assets = results_df['asset']
    total = len(assets)
    for a in ['UPRO', 'QQQ', 'SHY']:
        count = (assets == a).sum()
        pct = count / total
        # Compute return while in this asset
        mask = assets == a
        if mask.sum() > 0:
            m = compute_metrics(results_df.loc[mask, 'return'], a)
            print(f"  {a:6s}: {count:>5d} days ({pct:>5.1%})  "
                  f"Sharpe={m['sharpe']:+.2f}  Ann.Ret={m['cagr']:+.1%}")

    # Switching stats
    switches = (assets != assets.shift(1)).sum() - 1  # Exclude first day
    print(f"\n  Total switches: {switches}")
    total_cost = results_df['cost'].sum()
    print(f"  Total cost drag: {total_cost:.4f} ({total_cost*100:.2f}%)")
    cb_days = results_df['circuit_breaker'].sum()
    print(f"  Circuit breaker days: {cb_days} ({cb_days/total:.1%})")


def worst_drawdowns(results_df, n=5):
    print(f"\n" + "="*80)
    print(f"TOP {n} DRAWDOWN EPISODES")
    print("="*80)

    eq = results_df['equity']
    hwm = eq.cummax()
    dd = (eq - hwm) / hwm

    episodes = []
    start = None
    for date, val in dd.items():
        if val < 0 and start is None:
            start = date
        elif val >= 0 and start is not None:
            ep_dd = dd[start:date]
            episodes.append({
                'start': start, 'end': date, 'trough': ep_dd.idxmin(),
                'max_dd': ep_dd.min(), 'days': (date - start).days
            })
            start = None
    if start:
        ep_dd = dd[start:]
        episodes.append({
            'start': start, 'end': dd.index[-1], 'trough': ep_dd.idxmin(),
            'max_dd': ep_dd.min(), 'days': (dd.index[-1] - start).days
        })

    episodes.sort(key=lambda x: x['max_dd'])
    for i, ep in enumerate(episodes[:n]):
        print(f"  #{i+1}: {ep['max_dd']:.1%} DD  |  "
              f"{ep['start'].strftime('%Y-%m-%d')} -> {ep['trough'].strftime('%Y-%m-%d')} -> {ep['end'].strftime('%Y-%m-%d')}  |  "
              f"{ep['days']}d")


def monthly_distribution(results_df):
    print("\n" + "="*80)
    print("MONTHLY RETURN DISTRIBUTION")
    print("="*80)

    monthly = results_df['return'].resample('ME').apply(lambda x: (1+x).prod()-1)
    print(f"  N months:    {len(monthly)}")
    print(f"  Mean:        {monthly.mean():.2%}")
    print(f"  Median:      {monthly.median():.2%}")
    print(f"  Std:         {monthly.std():.2%}")
    print(f"  Best:        {monthly.max():.2%} ({monthly.idxmax().strftime('%Y-%m')})")
    print(f"  Worst:       {monthly.min():.2%} ({monthly.idxmin().strftime('%Y-%m')})")
    print(f"  % positive:  {(monthly > 0).mean():.1%}")


def no_weekly_throttle_test(prices, signals):
    """Test daily rebalancing vs weekly to see throttle impact."""
    print("\n" + "="*80)
    print("REBALANCE FREQUENCY SENSITIVITY")
    print("="*80)

    res_weekly = run_backtest(prices, signals, weekly_only=True)
    m_w = compute_metrics(res_weekly['return'], 'Weekly')

    res_daily = run_backtest(prices, signals, weekly_only=False)
    m_d = compute_metrics(res_daily['return'], 'Daily')

    print(f"  Weekly rebal: Sharpe={m_w['sharpe']:.3f}  CAGR={m_w['cagr']:.1%}  Switches={((res_weekly['asset'] != res_weekly['asset'].shift(1)).sum()-1)}")
    print(f"  Daily rebal:  Sharpe={m_d['sharpe']:.3f}  CAGR={m_d['cagr']:.1%}  Switches={((res_daily['asset'] != res_daily['asset'].shift(1)).sum()-1)}")


def main():
    print("="*80)
    print("ADAPTIVE LEVERAGED GROWTH v2 — VIX-Overlay UPRO")
    print("="*80)
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Eval: {CONFIG['eval_start']} to {CONFIG['end_date']}")
    print()

    prices = download_data()
    print("\nComputing signals (3 signals, all T-1)...")
    signals = compute_signals(prices)

    # Signal stats
    eval_mask = prices.index >= pd.Timestamp(CONFIG['eval_start'])
    print(f"  above_sma: {signals.loc[eval_mask, 'above_sma'].mean():.1%} bullish")
    print(f"  vix_calm (level=2): {(signals.loc[eval_mask, 'vix_level'] == 2).mean():.1%}")
    print(f"  vix_moderate (level=1): {(signals.loc[eval_mask, 'vix_level'] == 1).mean():.1%}")
    print(f"  vix_high (level=0): {(signals.loc[eval_mask, 'vix_level'] == 0).mean():.1%}")
    print(f"  vix_contango: {signals.loc[eval_mask, 'vix_contango'].mean():.1%}")

    # Main backtest
    print("\n" + "="*80)
    print("MAIN BACKTEST (weekly rebalance)")
    print("="*80)

    results = run_backtest(prices, signals, use_lag=True, weekly_only=True)
    main_m = compute_metrics(results['return'], 'VIX-Overlay UPRO v2')

    print(f"\n  CAGR:           {main_m['cagr']:.1%}")
    print(f"  Volatility:     {main_m['volatility']:.1%}")
    print(f"  Sharpe:         {main_m['sharpe']:.3f}")
    print(f"  Sortino:        {main_m['sortino']:.3f}")
    print(f"  Max Drawdown:   {main_m['max_drawdown']:.1%}")
    print(f"  Calmar:         {main_m['calmar']:.3f}")
    print(f"  Win Rate:       {main_m['win_rate']:.1%}")
    print(f"  Profit Factor:  {main_m['profit_factor']:.3f}")
    print(f"  Total Return:   {main_m['total_return']:.0%}")
    print(f"  N days:         {main_m['n_days']}")

    # Allocation stats
    allocation_stats(results)

    # Benchmarks
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON")
    print("="*80)

    benchmarks = run_benchmarks(prices)
    all_metrics = [main_m]
    for bname, brets in benchmarks.items():
        all_metrics.append(compute_metrics(brets, bname))
    print_comparison(all_metrics)

    # Year by year
    year_by_year(results, benchmarks)

    # Regime analysis
    regime_analysis(results, prices)

    # Lag sensitivity
    m_t1, m_t0, lag_ratio = lag_sensitivity(prices, signals)

    # Rebalance frequency
    no_weekly_throttle_test(prices, signals)

    # Permutation test
    real_sharpe, perm_sharpes, p_value = permutation_test(prices, signals, CONFIG['n_perms'])

    # Monthly distribution
    monthly_distribution(results)

    # Worst drawdowns
    worst_drawdowns(results)

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    spy_sharpe = compute_metrics(benchmarks.get('SPY B&H', pd.Series()), 'SPY').get('sharpe', 0)
    upro_sma_sharpe = compute_metrics(benchmarks.get('UPRO+200SMA', pd.Series()), 'U+SMA').get('sharpe', 0)
    upro_sma_dd = compute_metrics(benchmarks.get('UPRO+200SMA', pd.Series()), 'U+SMA').get('max_drawdown', -1)

    checks = {
        'Sharpe > 0.5': main_m['sharpe'] > 0.5,
        'Sharpe > SPY B&H': main_m['sharpe'] > spy_sharpe,
        'MaxDD better than UPRO+200SMA': main_m['max_drawdown'] > upro_sma_dd,
        'MaxDD < -50%': main_m['max_drawdown'] > -0.50,
        'Perm test p < 0.05': p_value < 0.05,
        'T-0/T-1 ratio < 2.0': lag_ratio < 2.0,
        'OOS days >= 100': main_m['n_days'] >= 100,
        'Win rate > 50%': main_m['win_rate'] > 0.50,
        'Calmar > 0.3': main_m['calmar'] > 0.3,
    }

    for check, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")

    n_pass = sum(checks.values())
    n_total = len(checks)
    print(f"\n  Score: {n_pass}/{n_total}")

    if n_pass >= 7:
        print("  VERDICT: STRONG — ready for paper trading")
    elif n_pass >= 5:
        print("  VERDICT: PROMISING — needs parameter tuning")
    else:
        print("  VERDICT: WEAK — fundamental approach may not work")

    # Key comparison vs UPRO+200SMA
    print(f"\n  vs UPRO+200SMA baseline:")
    print(f"    Sharpe:  {main_m['sharpe']:.3f} vs {upro_sma_sharpe:.3f} "
          f"({'BETTER' if main_m['sharpe'] > upro_sma_sharpe else 'WORSE'})")
    print(f"    MaxDD:   {main_m['max_drawdown']:.1%} vs {upro_sma_dd:.1%} "
          f"({'BETTER' if main_m['max_drawdown'] > upro_sma_dd else 'WORSE'})")

    # Save
    output = {
        'strategy': 'VIX-Overlay UPRO v2',
        'run_time': datetime.now().isoformat(),
        'metrics': main_m,
        'checks': {k: bool(v) for k, v in checks.items()},
        'perm_p_value': float(p_value),
        'lag_ratio': float(lag_ratio),
    }

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    out_path = RESULTS_DIR / 'adaptive_leveraged_growth_v2_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=convert)

    print(f"\n  Results saved to {out_path}")
    return output


if __name__ == '__main__':
    main()
