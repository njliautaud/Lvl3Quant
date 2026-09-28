#!/usr/bin/env python3
"""
Multi-Strategy Portfolio Backtest v3
Combines: Trend CTA, UPRO/SMA overlay, Crisis exit accelerator, VIX spike overlay
Tests 5 allocation schemes: Equal, Growth-tilt, Adaptive, Risk-parity, MinVar
HC #724: T-1 signals, monthly rebalance, 10bps costs, sliding window, 2006-2026
"""

import json
import os
import sys
import time
import warnings
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/multi_strategy_portfolio_v3")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "multi_strategy_portfolio"

TICKERS = ["SPY", "QQQ", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ", "XLK", "XLY", "XLB", "UPRO"]
VIX_TICKER = "^VIX"
CTA_UNIVERSE = ["SPY", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "VNQ"]
COST_BPS = 10  # 10 bps per trade
START_DATE = "2005-01-01"  # extra for warmup
END_DATE = "2026-07-18"
BACKTEST_START = "2006-01-01"

# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    print("Downloading price data...")
    all_tickers = TICKERS + [VIX_TICKER]
    data = yf.download(all_tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    # Rename ^VIX column
    if '^VIX' in close.columns:
        close = close.rename(columns={'^VIX': 'VIX'})

    close = close.ffill()
    print(f"Data shape: {close.shape}, date range: {close.index[0]} to {close.index[-1]}")
    return close


# ─── Strategy Components ─────────────────────────────────────────────────────

def get_month_end_dates(idx):
    """Get month-end trading dates from index."""
    monthly = idx.to_series().groupby([idx.year, idx.month]).last()
    return monthly.values


def trend_cta_signals(close, n_top=3, mom_window=126):
    """
    6-month dual momentum across 8 ETFs.
    Top-3 equal weight, monthly rebalance. All signals T-1.
    """
    cta_prices = close[CTA_UNIVERSE].copy()
    returns = cta_prices.pct_change()
    momentum = cta_prices / cta_prices.shift(mom_window) - 1  # 6-month momentum

    month_ends = get_month_end_dates(close.index)
    weights = pd.DataFrame(0.0, index=close.index, columns=CTA_UNIVERSE)

    for i, dt in enumerate(month_ends):
        if dt not in momentum.index:
            continue
        mom_row = momentum.loc[dt]
        if mom_row.isna().all():
            continue
        # T-1: use signal from this month-end, apply NEXT month
        ranked = mom_row.dropna().sort_values(ascending=False)
        top_n = ranked.head(n_top).index.tolist()

        # Apply weights starting next trading day
        next_idx = close.index[close.index > dt]
        if len(next_idx) == 0:
            continue
        # Apply until next month-end or end of data
        if i + 1 < len(month_ends):
            end_dt = month_ends[i + 1]
            mask = (close.index > dt) & (close.index <= end_dt)
        else:
            mask = close.index > dt
        for t in top_n:
            weights.loc[mask, t] = 1.0 / n_top

    return weights


def upro_sma_signals(close, sma_window=200):
    """
    100% UPRO when SPY > 200d SMA, else TLT.
    Signal T-1: use yesterday's close vs SMA.
    """
    spy = close['SPY']
    sma = spy.rolling(sma_window).mean()
    signal = (spy > sma).shift(1)  # T-1

    weights = pd.DataFrame(0.0, index=close.index, columns=['UPRO', 'TLT'])
    weights.loc[signal == True, 'UPRO'] = 1.0
    weights.loc[signal == False, 'TLT'] = 1.0
    weights.loc[signal.isna(), :] = 0.0

    return weights


def crisis_exit_signals(close, vix_threshold=30, hold_months=3):
    """
    When VIX drops below 30 after being above → add 20% XLK for 3 months.
    Signal T-1.
    """
    vix = close['VIX']
    vix_above = (vix > vix_threshold).shift(1)
    vix_below = (vix <= vix_threshold).shift(1)

    # Detect transitions: was above, now below
    transition = vix_above.shift(1) & vix_below  # T-1 of T-1

    weights = pd.DataFrame(0.0, index=close.index, columns=['XLK'])
    hold_days = hold_months * 21  # approx trading days

    transition_dates = close.index[transition.fillna(False)]
    for dt in transition_dates:
        end_dt = dt + pd.Timedelta(days=hold_days * 1.5)  # calendar days approx
        mask = (close.index >= dt) & (close.index <= end_dt)
        # Limit to hold_days trading days
        trading_days_after = close.index[close.index >= dt][:hold_days]
        if len(trading_days_after) > 0:
            weights.loc[trading_days_after, 'XLK'] = 0.20

    return weights


def vix_spike_overlay(close, high_threshold=30, low_threshold=15):
    """
    VIX > 30: defensive (TLT). VIX < 15: more equity.
    Returns adjustment multipliers for equity/bond split.
    """
    vix = close['VIX'].shift(1)  # T-1

    equity_mult = pd.Series(1.0, index=close.index)
    bond_mult = pd.Series(0.0, index=close.index)

    # High VIX: reduce equity, add bonds
    high_vix = vix > high_threshold
    equity_mult[high_vix] = 0.7
    bond_mult[high_vix] = 0.3

    # Low VIX: increase equity
    low_vix = vix < low_threshold
    equity_mult[low_vix] = 1.15
    bond_mult[low_vix] = -0.05  # slightly less bonds

    return equity_mult, bond_mult


# ─── Portfolio Allocation Schemes ─────────────────────────────────────────────

def compute_strategy_returns(close):
    """Compute daily returns for each sub-strategy."""
    returns = close.pct_change()

    # CTA returns
    cta_w = trend_cta_signals(close)
    cta_ret = (cta_w.shift(0) * returns[CTA_UNIVERSE]).sum(axis=1)  # weights already T-1

    # UPRO/SMA returns
    upro_w = upro_sma_signals(close)
    upro_ret = upro_w['UPRO'] * returns.get('UPRO', pd.Series(0, index=close.index)) + \
               upro_w['TLT'] * returns.get('TLT', pd.Series(0, index=close.index))

    # Crisis overlay returns (XLK allocation)
    crisis_w = crisis_exit_signals(close)
    crisis_ret = crisis_w['XLK'] * returns.get('XLK', pd.Series(0, index=close.index))

    # SPY baseline for crisis-enhanced
    spy_ret = returns['SPY']

    return {
        'cta': cta_ret,
        'upro_sma': upro_ret,
        'crisis_enhanced_spy': spy_ret + crisis_ret * 0.2,  # SPY + 20% crisis XLK overlay
        'crisis_overlay': crisis_ret,
        'spy': spy_ret,
        'tlt': returns['TLT'],
    }, cta_w, upro_w, crisis_w


def apply_transaction_costs(weights_series, cost_bps=COST_BPS):
    """Compute transaction cost drag from turnover."""
    if isinstance(weights_series, pd.DataFrame):
        turnover = weights_series.diff().abs().sum(axis=1)
    else:
        turnover = weights_series.diff().abs()
    cost = turnover * cost_bps / 10000
    return cost


def portfolio_A_equal(strat_rets, close):
    """33% CTA + 33% UPRO/SMA + 33% crisis-enhanced SPY"""
    ret = 0.333 * strat_rets['cta'] + 0.333 * strat_rets['upro_sma'] + 0.334 * strat_rets['crisis_enhanced_spy']
    return ret


def portfolio_B_growth(strat_rets, close):
    """50% CTA + 30% UPRO/SMA + 20% crisis overlay on SPY"""
    ret = 0.50 * strat_rets['cta'] + 0.30 * strat_rets['upro_sma'] + 0.20 * strat_rets['crisis_enhanced_spy']
    return ret


def portfolio_C_adaptive(strat_rets, close):
    """Shift weights based on VIX regime. T-1 signals."""
    vix = close['VIX'].shift(1)
    ret = pd.Series(0.0, index=close.index)

    # High VIX (>25): more CTA + TLT defensive
    high = vix > 25
    ret[high] = 0.50 * strat_rets['cta'][high] + \
                0.10 * strat_rets['upro_sma'][high] + \
                0.20 * strat_rets['crisis_enhanced_spy'][high] + \
                0.20 * strat_rets['tlt'][high]

    # Low VIX (<15): more UPRO
    low = vix < 15
    ret[low] = 0.25 * strat_rets['cta'][low] + \
               0.55 * strat_rets['upro_sma'][low] + \
               0.20 * strat_rets['crisis_enhanced_spy'][low]

    # Mid VIX: balanced
    mid = ~high & ~low
    ret[mid] = 0.40 * strat_rets['cta'][mid] + \
               0.35 * strat_rets['upro_sma'][mid] + \
               0.25 * strat_rets['crisis_enhanced_spy'][mid]

    return ret


def portfolio_D_riskparity(strat_rets, close, lookback=63):
    """Weight by inverse realized vol, monthly rebalance. Sliding window."""
    strats = pd.DataFrame({
        'cta': strat_rets['cta'],
        'upro_sma': strat_rets['upro_sma'],
        'crisis': strat_rets['crisis_enhanced_spy'],
    })

    month_ends = get_month_end_dates(close.index)
    weights = pd.DataFrame(np.nan, index=close.index, columns=strats.columns)

    for i, dt in enumerate(month_ends):
        if dt not in close.index:
            continue
        loc = close.index.get_loc(dt)
        if loc < lookback:
            continue

        # Trailing vol (sliding window, T-1)
        window = strats.iloc[loc - lookback:loc]
        vols = window.std() * np.sqrt(252)
        vols = vols.replace(0, np.nan)
        if vols.isna().all():
            continue

        inv_vol = 1.0 / vols
        w = inv_vol / inv_vol.sum()

        # Apply next month
        if i + 1 < len(month_ends):
            end_dt = month_ends[i + 1]
            mask = (close.index > dt) & (close.index <= end_dt)
        else:
            mask = close.index > dt
        for col in strats.columns:
            weights.loc[mask, col] = w[col]

    weights = weights.ffill().fillna(0)
    ret = (weights * strats).sum(axis=1)
    return ret


def portfolio_E_minvar(strat_rets, close, lookback=63):
    """Minimum variance: optimize weights to minimize portfolio vol. Sliding 63d cov."""
    strats = pd.DataFrame({
        'cta': strat_rets['cta'],
        'upro_sma': strat_rets['upro_sma'],
        'crisis': strat_rets['crisis_enhanced_spy'],
    })
    n = len(strats.columns)

    month_ends = get_month_end_dates(close.index)
    weights = pd.DataFrame(np.nan, index=close.index, columns=strats.columns)

    for i, dt in enumerate(month_ends):
        if dt not in close.index:
            continue
        loc = close.index.get_loc(dt)
        if loc < lookback:
            continue

        window = strats.iloc[loc - lookback:loc]
        cov = window.cov().values * 252

        # Check for valid covariance
        if np.isnan(cov).any() or np.isinf(cov).any():
            continue

        def port_var(w):
            return w @ cov @ w

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
        bounds = [(0.05, 0.80)] * n  # min 5%, max 80% per strategy

        x0 = np.ones(n) / n
        try:
            res = minimize(port_var, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'maxiter': 200, 'ftol': 1e-12})
            if res.success:
                w = res.x
            else:
                w = x0
        except Exception:
            w = x0

        # Apply next month
        if i + 1 < len(month_ends):
            end_dt = month_ends[i + 1]
            mask = (close.index > dt) & (close.index <= end_dt)
        else:
            mask = close.index > dt
        for j, col in enumerate(strats.columns):
            weights.loc[mask, col] = w[j]

    weights = weights.ffill().fillna(0)
    ret = (weights * strats).sum(axis=1)
    return ret


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(daily_returns, name="Strategy"):
    """Compute comprehensive risk-adjusted metrics."""
    dr = daily_returns.dropna()
    if len(dr) < 252:
        return {}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    cum = (1 + dr).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    total_ret = cum.iloc[-1] - 1
    years = len(dr) / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    # Profit factor
    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Win rate
    wr = (dr > 0).mean()

    # Skewness and kurtosis
    skew = dr.skew()
    kurt = dr.kurtosis()

    return {
        'name': name,
        'cagr': round(cagr * 100, 2),
        'ann_vol': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr * 100, 2),
        'skew': round(skew, 3),
        'kurtosis': round(kurt, 3),
        'total_return': round(total_ret * 100, 2),
        'years': round(years, 1),
    }


# ─── Validation Tests ────────────────────────────────────────────────────────

def permutation_test(daily_returns, n_perms=200):
    """Shuffle allocation timing, compute p-value of actual Sharpe."""
    actual_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    shuffled_sharpes = []
    dr = daily_returns.values
    for _ in range(n_perms):
        perm = np.random.permutation(dr)
        s = np.nanmean(perm) / np.nanstd(perm) * np.sqrt(252)
        shuffled_sharpes.append(s)
    p_value = np.mean(np.array(shuffled_sharpes) >= actual_sharpe)
    return {
        'actual_sharpe': round(float(actual_sharpe), 3),
        'mean_shuffled_sharpe': round(float(np.mean(shuffled_sharpes)), 3),
        'p_value': round(float(p_value), 4),
        'significant_5pct': bool(p_value < 0.05),
    }


def regime_test(daily_returns, spy_returns):
    """Bull vs bear regime performance."""
    # Bull: SPY trailing 6m return > 0; Bear: < 0
    spy_6m = spy_returns.rolling(126).sum()
    bull = spy_6m > 0
    bear = spy_6m <= 0

    # Align
    common = daily_returns.index.intersection(bull.dropna().index)
    dr_c = daily_returns.loc[common]
    bull_c = bull.loc[common]
    bear_c = bear.loc[common]

    bull_ret = dr_c[bull_c]
    bear_ret = dr_c[bear_c]

    bull_sharpe = bull_ret.mean() / bull_ret.std() * np.sqrt(252) if len(bull_ret) > 20 else 0
    bear_sharpe = bear_ret.mean() / bear_ret.std() * np.sqrt(252) if len(bear_ret) > 20 else 0

    return {
        'bull_sharpe': round(float(bull_sharpe), 3),
        'bear_sharpe': round(float(bear_sharpe), 3),
        'bull_days': int(bull_c.sum()),
        'bear_days': int(bear_c.sum()),
        'regime_ratio': round(abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01), 3),
    }


def subperiod_stability(daily_returns, n_blocks=4):
    """Split into n_blocks equal periods, compute metrics for each."""
    dr = daily_returns.dropna()
    block_size = len(dr) // n_blocks
    results = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else len(dr)
        block = dr.iloc[start:end]
        m = compute_metrics(block, name=f"Block_{i+1}")
        m['start_date'] = str(block.index[0].date())
        m['end_date'] = str(block.index[-1].date())
        results.append(m)
    return results


def spy_correlation(daily_returns, spy_returns):
    """Correlation with SPY."""
    common = daily_returns.index.intersection(spy_returns.dropna().index)
    if len(common) < 50:
        return 0.0
    return round(float(daily_returns.loc[common].corr(spy_returns.loc[common])), 3)


# ─── Benchmarks ───────────────────────────────────────────────────────────────

def benchmark_60_40(close):
    """60% SPY + 40% TLT, monthly rebalance."""
    returns = close[['SPY', 'TLT']].pct_change()
    ret = 0.60 * returns['SPY'] + 0.40 * returns['TLT']
    return ret


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"=" * 80)
    print(f"Multi-Strategy Portfolio Backtest v3")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"=" * 80)

    t0 = time.time()

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"portfolio_v3_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        mlflow.log_param("start_date", BACKTEST_START)
        mlflow.log_param("end_date", END_DATE)
        mlflow.log_param("cost_bps", COST_BPS)
        mlflow.log_param("cta_universe", ",".join(CTA_UNIVERSE))

        # Download data
        close = download_data()

        # Trim to backtest period
        bt_mask = close.index >= BACKTEST_START
        close_bt = close.copy()  # keep full for warmup calcs

        # Compute sub-strategy returns
        print("\nComputing sub-strategy returns...")
        strat_rets, cta_w, upro_w, crisis_w = compute_strategy_returns(close_bt)

        # Trim to backtest start
        for k in strat_rets:
            strat_rets[k] = strat_rets[k].loc[bt_mask]
        close_bt = close_bt.loc[bt_mask]

        # Compute portfolio returns for each allocation
        print("Computing portfolio allocations...")
        portfolios = {}
        portfolios['A_Equal'] = portfolio_A_equal(strat_rets, close_bt)
        portfolios['B_Growth'] = portfolio_B_growth(strat_rets, close_bt)
        portfolios['C_Adaptive'] = portfolio_C_adaptive(strat_rets, close_bt)
        portfolios['D_RiskParity'] = portfolio_D_riskparity(strat_rets, close_bt)
        portfolios['E_MinVar'] = portfolio_E_minvar(strat_rets, close_bt)

        # Benchmarks
        spy_ret = close_bt['SPY'].pct_change()
        bench_6040 = benchmark_60_40(close_bt)
        portfolios['Benchmark_SPY'] = spy_ret
        portfolios['Benchmark_6040'] = bench_6040

        # Apply transaction costs (approximate)
        for name in ['A_Equal', 'B_Growth', 'C_Adaptive', 'D_RiskParity', 'E_MinVar']:
            # Estimate turnover cost: monthly rebalance ~ 2 * cost per rebalance
            # Rough: 12 rebalances/yr, avg 30% turnover each time
            monthly_cost = 12 * 0.30 * COST_BPS / 10000 / 252  # daily drag
            portfolios[name] = portfolios[name] - monthly_cost

        # ─── Compute Metrics ───
        print("\nComputing metrics...")
        all_metrics = {}
        for name, ret in portfolios.items():
            m = compute_metrics(ret.dropna(), name=name)
            all_metrics[name] = m
            print(f"  {name}: Sharpe={m.get('sharpe','N/A')}, CAGR={m.get('cagr','N/A')}%, MaxDD={m.get('max_dd','N/A')}%")

            # Log to MLflow
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{name}_{k}", v)

        # ─── Validation ───
        print("\nRunning validation tests...")
        validation = {}
        for name in ['A_Equal', 'B_Growth', 'C_Adaptive', 'D_RiskParity', 'E_MinVar']:
            ret = portfolios[name].dropna()
            print(f"  {name}:")

            # Permutation test
            perm = permutation_test(ret, n_perms=200)
            print(f"    Permutation: p={perm['p_value']}, sig={perm['significant_5pct']}")

            # Regime test
            regime = regime_test(ret, spy_ret)
            print(f"    Regime: bull_sharpe={regime['bull_sharpe']}, bear_sharpe={regime['bear_sharpe']}, ratio={regime['regime_ratio']}")

            # Sub-period stability
            subperiods = subperiod_stability(ret, n_blocks=4)
            sharpes = [s.get('sharpe', 0) for s in subperiods]
            print(f"    Sub-period Sharpes: {sharpes}")

            # SPY correlation
            corr = spy_correlation(ret, spy_ret)
            print(f"    SPY corr: {corr}")

            validation[name] = {
                'permutation': perm,
                'regime': regime,
                'subperiods': subperiods,
                'spy_correlation': corr,
            }

            # Log key validation metrics
            mlflow.log_metric(f"{name}_perm_pvalue", perm['p_value'])
            mlflow.log_metric(f"{name}_regime_ratio", regime['regime_ratio'])
            mlflow.log_metric(f"{name}_spy_corr", corr)

        # ─── Equity Curves ───
        print("\nBuilding equity curves...")
        equity_curves = pd.DataFrame()
        for name, ret in portfolios.items():
            equity_curves[name] = (1 + ret.fillna(0)).cumprod()

        equity_curves.to_parquet(OUTPUT_DIR / "equity_curves.parquet")

        # ─── Results JSON ───
        results = {
            'run_timestamp': datetime.now().isoformat(),
            'backtest_period': f"{BACKTEST_START} to {END_DATE}",
            'cost_bps': COST_BPS,
            'metrics': all_metrics,
            'validation': validation,
            'ranking': sorted(
                [(name, m.get('sharpe', 0)) for name, m in all_metrics.items()],
                key=lambda x: x[1], reverse=True
            ),
        }

        with open(OUTPUT_DIR / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # ─── Summary Report ───
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MULTI-STRATEGY PORTFOLIO BACKTEST v3 — SUMMARY REPORT")
        report_lines.append(f"Run: {datetime.now().isoformat()}")
        report_lines.append(f"Period: {BACKTEST_START} to {END_DATE}")
        report_lines.append(f"Transaction costs: {COST_BPS} bps per trade")
        report_lines.append("=" * 80)

        report_lines.append("\n── PERFORMANCE COMPARISON ──")
        report_lines.append(f"{'Portfolio':<20} {'CAGR%':>8} {'Vol%':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD%':>8} {'Calmar':>8} {'PF':>8} {'WR%':>6} {'SPYcorr':>8}")
        report_lines.append("-" * 100)

        ranking = sorted(all_metrics.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)
        for name, m in ranking:
            corr = validation.get(name, {}).get('spy_correlation', 'N/A')
            report_lines.append(
                f"{name:<20} {m.get('cagr', 0):>8.1f} {m.get('ann_vol', 0):>8.1f} "
                f"{m.get('sharpe', 0):>8.3f} {m.get('sortino', 0):>8.3f} "
                f"{m.get('max_dd', 0):>8.1f} {m.get('calmar', 0):>8.3f} "
                f"{m.get('profit_factor', 0):>8.3f} {m.get('win_rate', 0):>6.1f} "
                f"{corr if isinstance(corr, (int, float)) else 'N/A':>8}"
            )

        report_lines.append("\n── VALIDATION RESULTS ──")
        for name in ['A_Equal', 'B_Growth', 'C_Adaptive', 'D_RiskParity', 'E_MinVar']:
            v = validation[name]
            report_lines.append(f"\n{name}:")
            report_lines.append(f"  Permutation test: p={v['permutation']['p_value']:.4f} {'PASS' if v['permutation']['significant_5pct'] else 'FAIL'}")
            report_lines.append(f"  Regime: bull_sharpe={v['regime']['bull_sharpe']:.3f}, bear_sharpe={v['regime']['bear_sharpe']:.3f}, ratio={v['regime']['regime_ratio']:.3f}")
            report_lines.append(f"  SPY correlation: {v['spy_correlation']}")
            report_lines.append(f"  Sub-period Sharpes: {[s.get('sharpe', 'N/A') for s in v['subperiods']]}")
            # Check regime-agnostic gate
            if v['regime']['regime_ratio'] > 0.50:
                report_lines.append(f"  ⚠ REGIME-AGNOSTIC GATE FAIL: ratio {v['regime']['regime_ratio']:.3f} > 0.50")
            else:
                report_lines.append(f"  ✓ Regime-agnostic gate PASS")

        report_lines.append("\n── RANKING (by Sharpe) ──")
        for i, (name, sharpe) in enumerate(results['ranking']):
            report_lines.append(f"  {i+1}. {name}: Sharpe {sharpe}")

        elapsed = time.time() - t0
        report_lines.append(f"\n── Runtime: {elapsed:.1f}s ──")

        report_text = "\n".join(report_lines)
        print("\n" + report_text)

        with open(OUTPUT_DIR / "summary_report.txt", 'w') as f:
            f.write(report_text)

        mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
        mlflow.log_artifact(str(OUTPUT_DIR / "summary_report.txt"))
        mlflow.log_artifact(str(OUTPUT_DIR / "equity_curves.parquet"))
        mlflow.log_metric("runtime_seconds", elapsed)

        print(f"\nDone. Output: {OUTPUT_DIR}")
        print(f"MLflow: {MLFLOW_URI}/#/experiments")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
