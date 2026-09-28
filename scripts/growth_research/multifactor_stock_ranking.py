#!/usr/bin/env python3
"""
Multi-Factor Stock Ranking Backtest
====================================
Tests whether combining momentum, quality, value, and technical factors
can predict stock returns better than individual factors.

Method: Monthly rebalance, buy top quintile (20 stocks), compare vs
equal-weight benchmark and SPY.

Validation: Permutation test (100 shuffles), R1 regime test,
sub-period consistency, outlier removal.

Output: /home/jupiter/Lvl3Quant/output/growth_research/multifactor_ranking/
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
import yfinance as yf

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/multifactor_ranking'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# UNIVERSE: Top 100 most liquid S&P 500 names
# ============================================================
TOP_100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'JPM', 'V', 'XOM', 'PG', 'MA', 'HD', 'AVGO', 'CVX',
    'MRK', 'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'BAC', 'WMT', 'TMO',
    'CSCO', 'MCD', 'ABT', 'CRM', 'ACN', 'DHR', 'ADBE', 'NFLX', 'AMD',
    'TXN', 'CMCSA', 'NEE', 'PM', 'WFC', 'BMY', 'INTC', 'UPS', 'RTX',
    'ORCL', 'QCOM', 'HON', 'UNP', 'AMGN', 'LOW', 'SBUX', 'IBM', 'GE',
    'CAT', 'INTU', 'DE', 'BA', 'ISRG', 'GILD', 'GS', 'BLK', 'MDT',
    'SYK', 'ADP', 'MMM', 'MDLZ', 'BKNG', 'ADI', 'VRTX', 'REGN',
    'TJX', 'SCHW', 'PGR', 'CB', 'CI', 'SO', 'DUK', 'MO', 'ZTS',
    'LRCX', 'CME', 'BSX', 'EL', 'PLD', 'CL', 'SLB', 'EOG', 'SNPS',
    'APD', 'FIS', 'ICE', 'KLAC', 'USB', 'MCK', 'ANET', 'EMR', 'SHW',
    'PYPL', 'NOW', 'GD'
]

SPY_TICKER = 'SPY'


def download_data(tickers, start='2013-01-01', end='2026-07-15'):
    """Download price data for all tickers + SPY."""
    print(f"Downloading price data for {len(tickers)} tickers + SPY...")
    all_tickers = list(set(tickers + [SPY_TICKER]))

    data = yf.download(all_tickers, start=start, end=end,
                       auto_adjust=True, progress=False, threads=True)

    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume']
    else:
        close = data[['Close']].copy()
        volume = data[['Volume']].copy()

    # Drop tickers with too little data
    min_days = 252
    good_tickers = close.columns[close.count() >= min_days]
    close = close[good_tickers]
    volume = volume[[c for c in good_tickers if c in volume.columns]]

    print(f"Downloaded {len(close)} trading days, {close.shape[1]} tickers with sufficient data")
    return close, volume


def precompute_all_factors(close):
    """
    Vectorized precomputation of ALL factors for ALL dates at once.
    Returns a dict of DataFrames, each (dates x tickers).
    """
    print("Precomputing factors (vectorized)...")

    # Daily returns
    daily_rets = close.pct_change()

    tickers = [t for t in close.columns if t != SPY_TICKER]
    close_t = close[tickers]
    rets_t = daily_rets[tickers]

    factors = {}

    # ---- MOMENTUM ----
    # 12-month momentum (skip last month): price[-22] / price[-252] - 1
    factors['mom_12m'] = close_t.shift(22) / close_t.shift(252) - 1
    # 6-month momentum (skip last month)
    factors['mom_6m'] = close_t.shift(22) / close_t.shift(126) - 1

    # ---- VALUE (price-based proxies) ----
    # Price relative to 1-year average (higher = cheaper relative to avg)
    rolling_mean_252 = close_t.rolling(252).mean()
    factors['value_rel'] = rolling_mean_252 / close_t

    # Price relative to 52-week high (lower = further from high = "cheaper")
    rolling_max_252 = close_t.rolling(252).max()
    factors['pct_52wk_high'] = close_t / rolling_max_252

    # ---- QUALITY ----
    # Quality: low volatility of returns (negative std so lower vol = higher score)
    factors['quality_low_vol'] = -rets_t.rolling(252).std()

    # Quality: R-squared of cumulative returns (trend stability)
    # This is expensive to compute vectorized, use a rolling approach
    # Approximate: use rolling autocorrelation as proxy for trend stability
    factors['quality_stability'] = rets_t.rolling(63).mean() / rets_t.rolling(63).std()

    # Quality: trend strength (rolling mean return, annualized)
    factors['quality_trend'] = rets_t.rolling(252).mean() * 252

    # ---- TECHNICAL ----
    # RSI(14)
    delta = close_t.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    factors['rsi_14'] = 100 - (100 / (1 + rs))

    # SMA crossover: 50 SMA / 200 SMA
    sma50 = close_t.rolling(50).mean()
    sma200 = close_t.rolling(200).mean()
    factors['sma_crossover'] = sma50 / sma200

    # Volatility rank (negative so low vol = high rank)
    factors['vol_rank'] = -(rets_t.rolling(63).std() * np.sqrt(252))

    print(f"  Computed {len(factors)} factors for {len(tickers)} tickers")
    return factors


def get_factors_at_date(precomputed, date):
    """Extract factor values for a specific date from precomputed DataFrames."""
    factor_cols = list(precomputed.keys())

    # Find the exact date or nearest prior
    sample = precomputed[factor_cols[0]]
    valid_dates = sample.index[sample.index <= date]
    if len(valid_dates) == 0:
        return None
    actual_date = valid_dates[-1]

    rows = {}
    for fname, fdf in precomputed.items():
        rows[fname] = fdf.loc[actual_date]

    df = pd.DataFrame(rows)
    # Drop rows with all NaN
    df = df.dropna(how='all')
    return df


def zscore_cross_section(factor_df):
    """Cross-sectionally z-score all factors."""
    zscored = factor_df.copy()
    for col in zscored.columns:
        vals = zscored[col].dropna()
        if len(vals) > 5 and vals.std() > 0:
            zscored[col] = (zscored[col] - vals.mean()) / vals.std()
        else:
            zscored[col] = 0
    return zscored


def composite_score(factor_df):
    """Equal-weight z-score composite across all factors."""
    zscored = zscore_cross_section(factor_df)
    factor_cols = [c for c in zscored.columns if c in [
        'mom_12m', 'mom_6m', 'value_rel', 'pct_52wk_high',
        'quality_stability', 'quality_trend', 'quality_low_vol',
        'rsi_14', 'sma_crossover', 'vol_rank'
    ]]
    if not factor_cols:
        return pd.Series(dtype=float)
    scores = zscored[factor_cols].mean(axis=1)
    return scores.dropna().sort_values(ascending=False)


def compute_forward_returns(close, rebal_date, next_date):
    """Compute forward returns between two dates for all stocks."""
    # Price at rebalance date
    valid = close.index[close.index <= rebal_date]
    if len(valid) == 0:
        return pd.Series(dtype=float)
    start_prices = close.loc[valid[-1]]

    # Price at next rebalance
    valid_end = close.index[close.index <= next_date]
    if len(valid_end) == 0:
        return pd.Series(dtype=float)
    end_prices = close.loc[valid_end[-1]]

    fwd_ret = (end_prices / start_prices) - 1
    # Sanity filter
    fwd_ret = fwd_ret[(fwd_ret.abs() < 1.0) & fwd_ret.notna()]
    return fwd_ret


def run_backtest(close, precomputed, start_date='2015-01-01', end_date='2026-06-30',
                 n_top=20, shuffle_labels=False):
    """
    Monthly rebalance backtest using precomputed factors.
    """
    monthly_dates = close.resample('ME').last().index
    monthly_dates = monthly_dates[(monthly_dates >= start_date) & (monthly_dates <= end_date)]

    results = []

    for i in range(len(monthly_dates) - 1):
        rebal_date = monthly_dates[i]
        next_date = monthly_dates[i + 1]

        # Get factors at rebalance date
        factors = get_factors_at_date(precomputed, rebal_date)
        if factors is None or len(factors) < n_top * 2:
            continue

        # Composite score
        scores = composite_score(factors)

        if shuffle_labels:
            shuffled_idx = np.random.permutation(scores.index)
            scores = pd.Series(scores.values, index=shuffled_idx)

        if len(scores) < n_top:
            continue

        # Top quintile
        top_stocks = scores.head(n_top).index.tolist()

        # Forward returns
        fwd_rets = compute_forward_returns(close, rebal_date, next_date)
        if len(fwd_rets) < n_top:
            continue

        # Portfolio return (top quintile, equal weight)
        top_available = [s for s in top_stocks if s in fwd_rets.index]
        if len(top_available) < 5:
            continue
        port_ret = fwd_rets[top_available].mean()

        # Benchmark (equal weight all scored stocks)
        bench_stocks = [s for s in scores.index if s in fwd_rets.index and s != SPY_TICKER]
        bench_ret = fwd_rets[bench_stocks].mean() if bench_stocks else 0

        # SPY
        spy_ret = fwd_rets.get(SPY_TICKER, 0)
        if pd.isna(spy_ret):
            spy_ret = 0

        results.append({
            'date': next_date,
            'portfolio': port_ret,
            'benchmark': bench_ret,
            'spy': spy_ret,
            'n_stocks': len(top_available),
        })

    return pd.DataFrame(results)


def compute_metrics(returns_series, name='Strategy'):
    """Compute risk-adjusted metrics from monthly returns."""
    if len(returns_series) < 6:
        return {}

    ann_ret = (1 + returns_series).prod() ** (12 / len(returns_series)) - 1
    ann_vol = returns_series.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns_series[returns_series < 0]
    downside_vol = downside.std() * np.sqrt(12) if len(downside) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    cum = (1 + returns_series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    wr = (returns_series > 0).mean()

    gains = returns_series[returns_series > 0].sum()
    losses = abs(returns_series[returns_series < 0].sum())
    pf = gains / losses if losses > 0 else np.inf

    n_years = len(returns_series) / 12
    total_ret = (1 + returns_series).prod() - 1
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    return {
        'name': name,
        'CAGR': f"{cagr:.1%}",
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'MaxDD': f"{max_dd:.1%}",
        'WinRate': f"{wr:.1%}",
        'ProfitFactor': round(pf, 2),
        'AnnVol': f"{ann_vol:.1%}",
        'Months': len(returns_series),
        'TotalReturn': f"{total_ret:.1%}",
        'sharpe_raw': sharpe,
        'cagr_raw': cagr
    }


def permutation_test(close, precomputed, real_sharpe, n_perms=100):
    """Permutation test: shuffle scores, re-run backtest."""
    print(f"\nRunning permutation test ({n_perms} shuffles)...")
    perm_sharpes = []

    for i in range(n_perms):
        if (i + 1) % 25 == 0:
            print(f"  Permutation {i+1}/{n_perms}...")

        np.random.seed(i + 42)
        perm_results = run_backtest(close, precomputed, shuffle_labels=True)

        if len(perm_results) > 6:
            perm_rets = perm_results['portfolio']
            ann_ret = (1 + perm_rets).prod() ** (12 / len(perm_rets)) - 1
            ann_vol = perm_rets.std() * np.sqrt(12)
            perm_sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    print(f"  Real Sharpe: {real_sharpe:.3f}")
    print(f"  Perm mean:   {perm_sharpes.mean():.3f} +/- {perm_sharpes.std():.3f}")
    print(f"  Perm p-value: {p_value:.3f}")
    print(f"  VERDICT: {'PASS (p < 0.05)' if p_value < 0.05 else 'FAIL (p >= 0.05)'}")

    return p_value, perm_sharpes


def regime_test(results_df):
    """R1 regime-agnostic test."""
    spy_rets = results_df.set_index('date')['spy']
    port_rets = results_df.set_index('date')['portfolio']
    excess = port_rets - spy_rets

    bull_mask = spy_rets > 0
    bear_mask = spy_rets <= 0

    bull_excess = excess[bull_mask]
    bear_excess = excess[bear_mask]

    if len(bull_excess) < 3 or len(bear_excess) < 3:
        print("  Not enough data for regime test")
        return np.nan

    bull_sharpe = bull_excess.mean() / bull_excess.std() * np.sqrt(12) if bull_excess.std() > 0 else 0
    bear_sharpe = bear_excess.mean() / bear_excess.std() * np.sqrt(12) if bear_excess.std() > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    r1_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0

    print(f"\nR1 Regime Test:")
    print(f"  Bull months (SPY > 0):  excess Sharpe = {bull_sharpe:.2f} ({len(bull_excess)} months)")
    print(f"  Bear months (SPY <= 0): excess Sharpe = {bear_sharpe:.2f} ({len(bear_excess)} months)")
    print(f"  R1 Gap: {r1_gap:.2f}")
    print(f"  VERDICT: {'PASS (gap < 0.50)' if r1_gap < 0.50 else 'FAIL (gap >= 0.50)'}")

    return r1_gap


def subperiod_test(results_df):
    """Test consistency across sub-periods."""
    print(f"\nSub-Period Consistency Test:")

    results_df = results_df.copy()
    results_df['date'] = pd.to_datetime(results_df['date'])

    n = len(results_df)
    splits = [0, n // 3, 2 * n // 3, n]

    sub_sharpes = []
    sub_excess_sharpes = []

    for i in range(3):
        sub = results_df.iloc[splits[i]:splits[i+1]]
        if len(sub) < 6:
            continue

        port_rets = sub['portfolio']
        spy_rets = sub['spy']
        excess = port_rets - spy_rets

        period_start = sub['date'].iloc[0].strftime('%Y-%m')
        period_end = sub['date'].iloc[-1].strftime('%Y-%m')

        sharpe = port_rets.mean() / port_rets.std() * np.sqrt(12) if port_rets.std() > 0 else 0
        ex_sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0

        sub_sharpes.append(sharpe)
        sub_excess_sharpes.append(ex_sharpe)

        print(f"  Period {i+1} ({period_start} to {period_end}): "
              f"Sharpe={sharpe:.2f}, Excess Sharpe={ex_sharpe:.2f}, "
              f"Months={len(sub)}")

    if len(sub_sharpes) < 3:
        print("  Not enough sub-periods")
        return False

    all_positive = all(s > 0 for s in sub_excess_sharpes)
    all_negative = all(s < 0 for s in sub_excess_sharpes)
    consistent = all_positive or all_negative

    cv = np.std(sub_sharpes) / abs(np.mean(sub_sharpes)) if abs(np.mean(sub_sharpes)) > 0 else np.inf

    print(f"  Sharpe CV across periods: {cv:.2f}")
    print(f"  Excess Sharpe signs consistent: {consistent}")
    print(f"  VERDICT: {'PASS' if consistent and cv < 2.0 else 'FAIL'}")

    return consistent and cv < 2.0


def outlier_test(results_df):
    """Remove top 5 best months and re-check."""
    print(f"\nOutlier Removal Test:")

    excess = results_df['portfolio'] - results_df['spy']
    orig_sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0

    sorted_idx = excess.sort_values(ascending=False).index
    trimmed = excess.drop(sorted_idx[:5])
    trim_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(12) if trimmed.std() > 0 else 0

    double_trimmed = excess.drop(sorted_idx[:5]).drop(sorted_idx[-5:], errors='ignore')
    dt_sharpe = double_trimmed.mean() / double_trimmed.std() * np.sqrt(12) if double_trimmed.std() > 0 else 0

    print(f"  Original excess Sharpe: {orig_sharpe:.2f}")
    print(f"  Remove 5 best months:  {trim_sharpe:.2f}")
    print(f"  Remove 5 best + 5 worst: {dt_sharpe:.2f}")

    retained = trim_sharpe / orig_sharpe if orig_sharpe != 0 else 0
    print(f"  Retained after trim: {retained:.0%}")
    print(f"  VERDICT: {'PASS (>50% retained)' if retained > 0.5 else 'FAIL (<50% retained)'}")

    return retained > 0.5


def individual_factor_analysis(close, precomputed, results_df):
    """Analyze individual factor ICs."""
    print(f"\n{'='*60}")
    print("INDIVIDUAL FACTOR ANALYSIS")
    print(f"{'='*60}")

    monthly_dates = close.resample('ME').last().index
    monthly_dates = monthly_dates[(monthly_dates >= '2015-01-01') & (monthly_dates <= '2026-06-30')]

    factor_names = list(precomputed.keys())
    factor_ics = {f: [] for f in factor_names}

    for i in range(len(monthly_dates) - 1):
        rebal_date = monthly_dates[i]
        next_date = monthly_dates[i + 1]

        factors = get_factors_at_date(precomputed, rebal_date)
        if factors is None or len(factors) < 20:
            continue

        fwd_rets = compute_forward_returns(close, rebal_date, next_date)
        if len(fwd_rets) < 20:
            continue

        for col in factor_names:
            if col not in factors.columns:
                continue
            common = factors[col].dropna().index.intersection(fwd_rets.index)
            common = [c for c in common if c != SPY_TICKER]
            if len(common) >= 20:
                ic = stats.spearmanr(factors.loc[common, col], fwd_rets[common])[0]
                if not np.isnan(ic):
                    factor_ics[col].append(ic)

    print(f"\n{'Factor':<25} {'Mean IC':>10} {'IC IR':>10} {'t-stat':>10} {'Hit Rate':>10}")
    print("-" * 65)

    for factor, ics in sorted(factor_ics.items(), key=lambda x: abs(np.mean(x[1])) if x[1] else 0, reverse=True):
        if not ics:
            continue
        ics = np.array(ics)
        mean_ic = ics.mean()
        ic_std = ics.std()
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0
        t_stat = mean_ic / (ic_std / np.sqrt(len(ics))) if ic_std > 0 else 0
        hit_rate = (ics > 0).mean() if mean_ic > 0 else (ics < 0).mean()

        print(f"  {factor:<23} {mean_ic:>10.4f} {ic_ir:>10.3f} {t_stat:>10.2f} {hit_rate:>10.1%}")

    return factor_ics


def main():
    print("=" * 60)
    print("MULTI-FACTOR STOCK RANKING BACKTEST")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe: Top 100 S&P 500 by liquidity")
    print(f"Method: Monthly rebalance, top quintile (20 stocks)")
    print(f"Period: 2015-01 to 2026-06")
    print()

    # Download data
    close, volume = download_data(TOP_100_TICKERS, start='2013-01-01', end='2026-07-15')

    # Precompute all factors (vectorized — fast)
    precomputed = precompute_all_factors(close)

    # Run main backtest
    print("\n" + "=" * 60)
    print("MAIN BACKTEST")
    print("=" * 60)

    results = run_backtest(close, precomputed)

    if len(results) < 12:
        print("ERROR: Not enough data for meaningful backtest")
        return

    # Compute metrics
    port_metrics = compute_metrics(results['portfolio'], 'Multi-Factor Top 20')
    bench_metrics = compute_metrics(results['benchmark'], 'Equal-Weight All')
    spy_metrics = compute_metrics(results['spy'], 'SPY')

    print(f"\n{'Metric':<20} {'Multi-Factor':>15} {'Equal-Wt All':>15} {'SPY':>15}")
    print("-" * 65)
    for key in ['CAGR', 'Sharpe', 'Sortino', 'MaxDD', 'WinRate', 'ProfitFactor', 'AnnVol', 'Months', 'TotalReturn']:
        print(f"  {key:<18} {port_metrics.get(key, 'N/A'):>15} {bench_metrics.get(key, 'N/A'):>15} {spy_metrics.get(key, 'N/A'):>15}")

    excess = results['portfolio'] - results['spy']
    excess_ann_ret = excess.mean() * 12
    excess_sharpe = excess.mean() / excess.std() * np.sqrt(12) if excess.std() > 0 else 0

    print(f"\n  Excess Return vs SPY (annualized): {excess_ann_ret:.1%}")
    print(f"  Excess Sharpe (Information Ratio):  {excess_sharpe:.2f}")

    real_sharpe = port_metrics['sharpe_raw']

    # ============================================================
    # VALIDATION GATE 1: Permutation Test
    # ============================================================
    print(f"\n{'='*60}")
    print("VALIDATION GATE 1: PERMUTATION TEST")
    print(f"{'='*60}")

    perm_p, perm_sharpes = permutation_test(close, precomputed, real_sharpe, n_perms=100)

    # ============================================================
    # VALIDATION GATE 2: R1 Regime Test
    # ============================================================
    print(f"\n{'='*60}")
    print("VALIDATION GATE 2: R1 REGIME TEST")
    print(f"{'='*60}")

    r1_gap = regime_test(results)

    # ============================================================
    # VALIDATION GATE 3: Sub-Period Consistency
    # ============================================================
    print(f"\n{'='*60}")
    print("VALIDATION GATE 3: SUB-PERIOD CONSISTENCY")
    print(f"{'='*60}")

    subperiod_pass = subperiod_test(results)

    # ============================================================
    # VALIDATION GATE 4: Outlier Removal
    # ============================================================
    print(f"\n{'='*60}")
    print("VALIDATION GATE 4: OUTLIER REMOVAL")
    print(f"{'='*60}")

    outlier_pass = outlier_test(results)

    # ============================================================
    # Individual Factor Analysis
    # ============================================================
    factor_ics = individual_factor_analysis(close, precomputed, results)

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print(f"\n{'='*60}")
    print("FINAL VERDICT")
    print(f"{'='*60}")

    gates = {
        'Permutation (p < 0.05)': perm_p < 0.05 if not np.isnan(perm_p) else False,
        'R1 Regime (gap < 0.50)': r1_gap < 0.50 if not np.isnan(r1_gap) else False,
        'Sub-Period Consistency': subperiod_pass,
        'Outlier Removal': outlier_pass
    }

    print(f"\n  Gate Results:")
    for gate, passed in gates.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"    {gate}: {status}")

    n_pass = sum(gates.values())
    n_total = len(gates)

    if n_pass == n_total:
        verdict = "VALIDATED — All gates passed. Multi-factor ranking shows real edge."
    elif n_pass >= 3:
        verdict = "CONDITIONAL — Passes most gates. Edge may exist but has caveats."
    elif n_pass >= 2:
        verdict = "WEAK — Passes some gates. Insufficient evidence of real edge."
    else:
        verdict = "REJECTED — Fails validation. No reliable edge over benchmarks."

    print(f"\n  Overall: {verdict}")
    print(f"\n  Key numbers:")
    print(f"    Multi-Factor Sharpe: {port_metrics['Sharpe']}")
    print(f"    SPY Sharpe:          {spy_metrics['Sharpe']}")
    print(f"    Excess Return:       {excess_ann_ret:.1%}/year")
    print(f"    Permutation p:       {perm_p:.3f}")
    print(f"    R1 Regime Gap:       {r1_gap:.2f}")

    # Save results
    results.to_csv(os.path.join(OUTPUT_DIR, 'monthly_returns.csv'), index=False)

    summary = {
        'strategy': 'Multi-Factor Stock Ranking',
        'universe': 'Top 100 S&P 500',
        'period': '2015-01 to 2026-06',
        'n_top': 20,
        'rebalance': 'Monthly',
        'portfolio_metrics': {k: v for k, v in port_metrics.items() if k not in ('sharpe_raw', 'cagr_raw')},
        'spy_metrics': {k: v for k, v in spy_metrics.items() if k not in ('sharpe_raw', 'cagr_raw')},
        'benchmark_metrics': {k: v for k, v in bench_metrics.items() if k not in ('sharpe_raw', 'cagr_raw')},
        'excess_return_annualized': f"{excess_ann_ret:.1%}",
        'information_ratio': round(excess_sharpe, 2),
        'validation': {
            'permutation_p': round(perm_p, 3) if not np.isnan(perm_p) else None,
            'r1_regime_gap': round(r1_gap, 2) if not np.isnan(r1_gap) else None,
            'subperiod_consistent': bool(subperiod_pass),
            'outlier_robust': bool(outlier_pass),
            'gates_passed': f"{n_pass}/{n_total}",
        },
        'verdict': verdict,
        'run_timestamp': datetime.now().isoformat()
    }

    def _convert(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=_convert)

    if len(perm_sharpes) > 0:
        np.save(os.path.join(OUTPUT_DIR, 'perm_sharpes.npy'), perm_sharpes)

    print(f"\n  Results saved to: {OUTPUT_DIR}/")
    print(f"  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == '__main__':
    main()
