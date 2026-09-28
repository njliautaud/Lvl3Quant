#!/usr/bin/env python3
"""
Pairs Trading / Statistical Arbitrage Backtest
6 variants across 6 pairs, with permutation tests and 5-gate validation.
Robinhood constraints: no shorting, $0 commission, $645 starting capital.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─── Config ───
START = '2020-01-01'
END = '2026-07-28'
OOT_START = '2022-01-01'
CAPITAL = 645.0
COMMISSION = 0.0
N_PERMS = 1000

PAIRS = [
    ('NVDA', 'AMD'),
    ('META', 'GOOGL'),
    ('UBER', 'LYFT'),
    ('MSFT', 'AAPL'),
    ('CRM', 'DDOG'),   # NOW may not be available on yfinance free
    ('AMZN', 'SHOP'),
]

# ─── Data Download ───
def download_data():
    tickers = set()
    for a, b in PAIRS:
        tickers.add(a)
        tickers.add(b)
    tickers.add('SPY')

    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(list(tickers), start=START, end=END, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    close = close.ffill().dropna(how='all')
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} rows")
    return close


def compute_spy_regime(close):
    """SPY 200-SMA regime: bull if above, bear if below."""
    spy = close['SPY']
    sma200 = spy.rolling(200).mean()
    regime = (spy > sma200).astype(int)  # 1=bull, 0=bear
    return regime


def compute_metrics(returns, regime, trades_list, capital=CAPITAL):
    """Compute all required metrics from a return series."""
    if len(returns) == 0 or returns.std() == 0:
        return _empty_metrics()

    # Align regime with returns
    regime_aligned = regime.reindex(returns.index).ffill()

    total_ret = (1 + returns).prod() - 1
    n_years = len(returns) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1 if total_ret > -1 else -1.0

    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * np.sqrt(252) if len(downside) > 0 and downside.std() > 0 else 0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    # Win rate and profit factor from trades
    if len(trades_list) > 0:
        wins = [t for t in trades_list if t > 0]
        losses = [t for t in trades_list if t < 0]
        win_rate = len(wins) / len(trades_list)
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1e-9
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0

    # Regime-stratified Sharpe
    bull_mask = regime_aligned == 1
    bear_mask = regime_aligned == 0

    bull_ret = returns[bull_mask]
    bear_ret = returns[bear_mask]

    sharpe_bull = bull_ret.mean() / bull_ret.std() * np.sqrt(252) if len(bull_ret) > 10 and bull_ret.std() > 0 else 0
    sharpe_bear = bear_ret.mean() / bear_ret.std() * np.sqrt(252) if len(bear_ret) > 10 and bear_ret.std() > 0 else 0

    max_s = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_s if max_s > 0 else 0

    return {
        'total_return_pct': round(total_ret * 100, 2),
        'cagr_pct': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(min(profit_factor, 99.99), 2),
        'total_trades': len(trades_list),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'regime_gap': round(regime_gap, 4),
    }


def _empty_metrics():
    return {
        'total_return_pct': 0, 'cagr_pct': 0, 'sharpe': 0, 'sortino': 0,
        'max_drawdown_pct': 0, 'win_rate': 0, 'profit_factor': 0, 'total_trades': 0,
        'sharpe_bull': 0, 'sharpe_bear': 0, 'regime_gap': 0,
    }


def permutation_test(returns, actual_sharpe, n_perms=N_PERMS):
    """Shuffle entry/exit timing. Return p-value."""
    if len(returns) < 20 or returns.std() == 0:
        return 1.0

    ret_vals = returns.values.copy()
    count_better = 0
    for _ in range(n_perms):
        np.random.shuffle(ret_vals)
        perm_sharpe = ret_vals.mean() / ret_vals.std() * np.sqrt(252)
        if perm_sharpe >= actual_sharpe:
            count_better += 1
    return count_better / n_perms


def five_gate(metrics, perm_p):
    """5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': metrics['regime_gap'] < 0.5,
        'mdd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'trades_gte_20': metrics['total_trades'] >= 20,
    }
    gates['pass_all'] = all(gates.values())
    return gates


# ═══════════════════════════════════════════════════
# VARIANT A: Classic Z-Score (long-leg only)
# ═══════════════════════════════════════════════════
def variant_a_classic_zscore(close, pair, regime, capital=CAPITAL):
    """
    Rolling 60-day z-score of log(price_A/price_B).
    z < -2: A is cheap relative to B -> buy A (long leg only).
    z > 2: B is cheap relative to A -> buy B (long leg only).
    Exit when z crosses 0.
    """
    a, b = pair
    if a not in close.columns or b not in close.columns:
        return pd.Series(dtype=float), []

    pa, pb = close[a], close[b]
    ratio = np.log(pa / pb)
    z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()

    oot_mask = close.index >= OOT_START
    z_oot = z[oot_mask]
    pa_oot = pa[oot_mask]
    pb_oot = pb[oot_mask]

    position = 0  # 0=flat, 1=long A, 2=long B
    entry_price = 0
    entry_ticker = None
    daily_returns = []
    trades = []

    for i in range(1, len(z_oot)):
        dt = z_oot.index[i]
        zval = z_oot.iloc[i]

        if pd.isna(zval):
            daily_returns.append(0)
            continue

        if position == 0:
            if zval < -2:
                # A is cheap -> buy A
                position = 1
                entry_price = pa_oot.iloc[i]
                entry_ticker = 'A'
                daily_returns.append(0)
            elif zval > 2:
                # B is cheap -> buy B
                position = 2
                entry_price = pb_oot.iloc[i]
                entry_ticker = 'B'
                daily_returns.append(0)
            else:
                daily_returns.append(0)
        else:
            if position == 1:
                ret = (pa_oot.iloc[i] - pa_oot.iloc[i-1]) / pa_oot.iloc[i-1]
            else:
                ret = (pb_oot.iloc[i] - pb_oot.iloc[i-1]) / pb_oot.iloc[i-1]
            daily_returns.append(ret)

            # Exit on z crossing 0
            prev_z = z_oot.iloc[i-1] if not pd.isna(z_oot.iloc[i-1]) else zval
            if (position == 1 and zval >= 0) or (position == 2 and zval <= 0):
                if position == 1:
                    pnl = (pa_oot.iloc[i] - entry_price) / entry_price
                else:
                    pnl = (pb_oot.iloc[i] - entry_price) / entry_price
                trades.append(pnl)
                position = 0
                entry_price = 0

    idx = z_oot.index[1:]
    returns = pd.Series(daily_returns, index=idx)
    return returns, trades


# ═══════════════════════════════════════════════════
# VARIANT B: Adaptive Bollinger Pairs
# ═══════════════════════════════════════════════════
def variant_b_bollinger(close, pair, regime, capital=CAPITAL):
    """
    20-day rolling mean and 2-sigma bands on price ratio.
    Dynamic position sizing based on distance from mean.
    """
    a, b = pair
    if a not in close.columns or b not in close.columns:
        return pd.Series(dtype=float), []

    pa, pb = close[a], close[b]
    ratio = pa / pb
    ma20 = ratio.rolling(20).mean()
    std20 = ratio.rolling(20).std()
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    oot_mask = close.index >= OOT_START
    ratio_oot = ratio[oot_mask]
    ma_oot = ma20[oot_mask]
    std_oot = std20[oot_mask]
    upper_oot = upper[oot_mask]
    lower_oot = lower[oot_mask]
    pa_oot = pa[oot_mask]
    pb_oot = pb[oot_mask]

    position = 0  # 0=flat, 1=long A (ratio below lower), 2=long B (ratio above upper)
    entry_price = 0
    size_frac = 0  # position size as fraction of capital
    daily_returns = []
    trades = []

    for i in range(1, len(ratio_oot)):
        if pd.isna(ma_oot.iloc[i]) or pd.isna(std_oot.iloc[i]) or std_oot.iloc[i] == 0:
            daily_returns.append(0)
            continue

        r = ratio_oot.iloc[i]
        m = ma_oot.iloc[i]
        s = std_oot.iloc[i]

        if position == 0:
            if r < lower_oot.iloc[i]:
                # Ratio low -> A is cheap -> buy A
                distance = (m - r) / s  # how many sigma below
                size_frac = min(distance / 4, 1.0)  # scale up to full size at 4 sigma
                position = 1
                entry_price = pa_oot.iloc[i]
                daily_returns.append(0)
            elif r > upper_oot.iloc[i]:
                # Ratio high -> B is cheap -> buy B
                distance = (r - m) / s
                size_frac = min(distance / 4, 1.0)
                position = 2
                entry_price = pb_oot.iloc[i]
                daily_returns.append(0)
            else:
                daily_returns.append(0)
        else:
            if position == 1:
                ret = (pa_oot.iloc[i] - pa_oot.iloc[i-1]) / pa_oot.iloc[i-1] * size_frac
            else:
                ret = (pb_oot.iloc[i] - pb_oot.iloc[i-1]) / pb_oot.iloc[i-1] * size_frac
            daily_returns.append(ret)

            # Exit at mean
            if (position == 1 and r >= m) or (position == 2 and r <= m):
                if position == 1:
                    pnl = (pa_oot.iloc[i] - entry_price) / entry_price * size_frac
                else:
                    pnl = (pb_oot.iloc[i] - entry_price) / entry_price * size_frac
                trades.append(pnl)
                position = 0

    idx = ratio_oot.index[1:]
    returns = pd.Series(daily_returns, index=idx)
    return returns, trades


# ═══════════════════════════════════════════════════
# VARIANT C: Multi-Pair Portfolio
# ═══════════════════════════════════════════════════
def variant_c_multipair(close, all_pairs, regime, capital=CAPITAL):
    """Run all 6 pairs simultaneously with equal allocation using Variant A logic."""
    all_returns = []
    all_trades = []

    for pair in all_pairs:
        ret, trades = variant_a_classic_zscore(close, pair, regime, capital / len(all_pairs))
        if len(ret) > 0:
            all_returns.append(ret)
            all_trades.extend(trades)

    if not all_returns:
        return pd.Series(dtype=float), []

    # Equal-weight combine
    combined = pd.concat(all_returns, axis=1).fillna(0)
    avg_ret = combined.mean(axis=1)
    return avg_ret, all_trades


# ═══════════════════════════════════════════════════
# VARIANT D: Momentum-Filtered Pairs
# ═══════════════════════════════════════════════════
def variant_d_momentum_filtered(close, pair, regime, capital=CAPITAL):
    """
    Only trade when spread is mean-reverting (not trending).
    Use autocorrelation of spread changes as Hurst proxy.
    Skip if autocorrelation > 0 for 20+ days (trending).
    """
    a, b = pair
    if a not in close.columns or b not in close.columns:
        return pd.Series(dtype=float), []

    pa, pb = close[a], close[b]
    ratio = np.log(pa / pb)
    z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()

    # Rolling autocorrelation of spread changes (Hurst proxy)
    spread_diff = ratio.diff()
    rolling_autocorr = spread_diff.rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) >= 10 else 0, raw=False
    )

    oot_mask = close.index >= OOT_START
    z_oot = z[oot_mask]
    pa_oot = pa[oot_mask]
    pb_oot = pb[oot_mask]
    ac_oot = rolling_autocorr[oot_mask]

    position = 0
    entry_price = 0
    daily_returns = []
    trades = []

    for i in range(1, len(z_oot)):
        zval = z_oot.iloc[i]
        ac = ac_oot.iloc[i] if not pd.isna(ac_oot.iloc[i]) else 0

        if pd.isna(zval):
            daily_returns.append(0)
            continue

        # Only enter if mean-reverting (negative autocorrelation)
        is_mean_reverting = ac < 0

        if position == 0:
            if is_mean_reverting and zval < -2:
                position = 1
                entry_price = pa_oot.iloc[i]
                daily_returns.append(0)
            elif is_mean_reverting and zval > 2:
                position = 2
                entry_price = pb_oot.iloc[i]
                daily_returns.append(0)
            else:
                daily_returns.append(0)
        else:
            if position == 1:
                ret = (pa_oot.iloc[i] - pa_oot.iloc[i-1]) / pa_oot.iloc[i-1]
            else:
                ret = (pb_oot.iloc[i] - pb_oot.iloc[i-1]) / pb_oot.iloc[i-1]
            daily_returns.append(ret)

            if (position == 1 and zval >= 0) or (position == 2 and zval <= 0):
                if position == 1:
                    pnl = (pa_oot.iloc[i] - entry_price) / entry_price
                else:
                    pnl = (pb_oot.iloc[i] - entry_price) / entry_price
                trades.append(pnl)
                position = 0

    idx = z_oot.index[1:]
    returns = pd.Series(daily_returns, index=idx)
    return returns, trades


# ═══════════════════════════════════════════════════
# VARIANT E: Cointegration-Based
# ═══════════════════════════════════════════════════
def variant_e_cointegration(close, pair, regime, capital=CAPITAL):
    """
    Rolling 60-day Engle-Granger cointegration test.
    Only trade when pair is cointegrated (p < 0.05).
    """
    a, b = pair
    if a not in close.columns or b not in close.columns:
        return pd.Series(dtype=float), []

    pa, pb = close[a], close[b]
    ratio = np.log(pa / pb)
    z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()

    # Precompute rolling cointegration p-values (expensive, so cache)
    coint_pvals = pd.Series(index=close.index, dtype=float)
    for i in range(60, len(pa)):
        try:
            # Engle-Granger: regress A on B, test residuals for unit root
            y = np.log(pa.iloc[i-60:i].values)
            x = np.log(pb.iloc[i-60:i].values)
            beta = np.polyfit(x, y, 1)[0]
            resid = y - beta * x
            adf_stat, adf_p, *_ = adfuller(resid, maxlag=5, autolag=None)
            coint_pvals.iloc[i] = adf_p
        except:
            coint_pvals.iloc[i] = 1.0

    oot_mask = close.index >= OOT_START
    z_oot = z[oot_mask]
    pa_oot = pa[oot_mask]
    pb_oot = pb[oot_mask]
    cp_oot = coint_pvals[oot_mask]

    position = 0
    entry_price = 0
    daily_returns = []
    trades = []

    for i in range(1, len(z_oot)):
        zval = z_oot.iloc[i]
        cp = cp_oot.iloc[i] if not pd.isna(cp_oot.iloc[i]) else 1.0

        if pd.isna(zval):
            daily_returns.append(0)
            continue

        is_cointegrated = cp < 0.05

        if position == 0:
            if is_cointegrated and zval < -2:
                position = 1
                entry_price = pa_oot.iloc[i]
                daily_returns.append(0)
            elif is_cointegrated and zval > 2:
                position = 2
                entry_price = pb_oot.iloc[i]
                daily_returns.append(0)
            else:
                daily_returns.append(0)
        else:
            if position == 1:
                ret = (pa_oot.iloc[i] - pa_oot.iloc[i-1]) / pa_oot.iloc[i-1]
            else:
                ret = (pb_oot.iloc[i] - pb_oot.iloc[i-1]) / pb_oot.iloc[i-1]
            daily_returns.append(ret)

            if (position == 1 and zval >= 0) or (position == 2 and zval <= 0):
                if position == 1:
                    pnl = (pa_oot.iloc[i] - entry_price) / entry_price
                else:
                    pnl = (pb_oot.iloc[i] - entry_price) / entry_price
                trades.append(pnl)
                position = 0

    idx = z_oot.index[1:]
    returns = pd.Series(daily_returns, index=idx)
    return returns, trades


# ═══════════════════════════════════════════════════
# VARIANT F: Options Pairs (BS-priced calls)
# ═══════════════════════════════════════════════════
def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def variant_f_options(close, pair, regime, capital=CAPITAL):
    """
    When z > 2.5 or z < -2.5 (extreme), buy a call on the cheap stock.
    30-45 DTE, ATM strike. IV=30%. Cap risk at $50/trade.
    """
    a, b = pair
    if a not in close.columns or b not in close.columns:
        return pd.Series(dtype=float), []

    pa, pb = close[a], close[b]
    ratio = np.log(pa / pb)
    z = (ratio - ratio.rolling(60).mean()) / ratio.rolling(60).std()

    IV = 0.30
    RISK_RATE = 0.05
    DTE_DAYS = 37  # ~37 calendar days
    MAX_RISK = 50.0

    oot_mask = close.index >= OOT_START
    z_oot = z[oot_mask]
    pa_oot = pa[oot_mask]
    pb_oot = pb[oot_mask]

    position = 0  # 0=flat, 1=holding call on A, 2=holding call on B
    call_entry_price = 0  # premium paid per contract
    n_contracts = 0
    strike = 0
    days_held = 0
    entry_dte = DTE_DAYS
    daily_returns = []
    trades = []

    for i in range(1, len(z_oot)):
        zval = z_oot.iloc[i]

        if pd.isna(zval):
            daily_returns.append(0)
            continue

        if position == 0:
            if zval < -2.5:
                # A is cheap -> buy call on A
                S = pa_oot.iloc[i]
                strike = round(S)  # ATM
                T = DTE_DAYS / 365
                premium = bs_call_price(S, strike, T, RISK_RATE, IV)
                if premium > 0:
                    n_contracts = min(int(MAX_RISK / (premium * 100)), 1)  # cap at $50 risk
                    if n_contracts < 1:
                        n_contracts = 1
                    actual_risk = premium * 100 * n_contracts
                    if actual_risk > MAX_RISK * 2:
                        daily_returns.append(0)
                        continue
                    call_entry_price = premium
                    position = 1
                    days_held = 0
                    entry_dte = DTE_DAYS
                daily_returns.append(0)
            elif zval > 2.5:
                # B is cheap -> buy call on B
                S = pb_oot.iloc[i]
                strike = round(S)
                T = DTE_DAYS / 365
                premium = bs_call_price(S, strike, T, RISK_RATE, IV)
                if premium > 0:
                    n_contracts = min(int(MAX_RISK / (premium * 100)), 1)
                    if n_contracts < 1:
                        n_contracts = 1
                    call_entry_price = premium
                    position = 2
                    days_held = 0
                    entry_dte = DTE_DAYS
                daily_returns.append(0)
            else:
                daily_returns.append(0)
        else:
            days_held += 1
            remaining_dte = max(entry_dte - days_held, 1)
            T = remaining_dte / 365

            if position == 1:
                S = pa_oot.iloc[i]
            else:
                S = pb_oot.iloc[i]

            current_premium = bs_call_price(S, strike, T, RISK_RATE, IV)
            prev_premium = call_entry_price if days_held == 1 else daily_returns[-1]  # approximate

            # Return based on option value change relative to capital risked
            option_ret = (current_premium - call_entry_price) / call_entry_price if call_entry_price > 0 else 0
            # Scale by fraction of capital at risk
            capital_frac = min(n_contracts * call_entry_price * 100 / capital, 1.0)
            day_ret = (current_premium - (call_entry_price if days_held == 1 else bs_call_price(
                pa_oot.iloc[i-1] if position == 1 else pb_oot.iloc[i-1],
                strike, (remaining_dte + 1) / 365, RISK_RATE, IV
            ))) / call_entry_price * capital_frac if call_entry_price > 0 else 0

            daily_returns.append(day_ret)

            # Exit: z crosses 0 or DTE < 5
            exit_signal = False
            if (position == 1 and zval >= 0) or (position == 2 and zval <= 0):
                exit_signal = True
            if remaining_dte <= 5:
                exit_signal = True

            if exit_signal:
                pnl = (current_premium - call_entry_price) * n_contracts * 100
                trades.append(pnl / capital)
                position = 0

    idx = z_oot.index[1:]
    returns = pd.Series(daily_returns, index=idx)
    return returns, trades


# ═══════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════
def main():
    close = download_data()
    regime = compute_spy_regime(close)

    results = {}

    # Variants that run per-pair
    variant_funcs = {
        'A_classic_zscore': variant_a_classic_zscore,
        'B_adaptive_bollinger': variant_b_bollinger,
        'D_momentum_filtered': variant_d_momentum_filtered,
        'E_cointegration_based': variant_e_cointegration,
        'F_options_pairs': variant_f_options,
    }

    for vname, vfunc in variant_funcs.items():
        print(f"\n{'='*60}")
        print(f"Running Variant {vname}")
        print(f"{'='*60}")

        variant_results = {}
        for pair in PAIRS:
            pair_name = f"{pair[0]}_{pair[1]}"
            print(f"  Processing {pair_name}...")

            returns, trades = vfunc(close, pair, regime)

            if len(returns) == 0:
                variant_results[pair_name] = {
                    'metrics': _empty_metrics(),
                    'perm_p': 1.0,
                    'five_gate': five_gate(_empty_metrics(), 1.0),
                }
                continue

            metrics = compute_metrics(returns, regime, trades)

            print(f"    Sharpe={metrics['sharpe']}, Trades={metrics['total_trades']}, "
                  f"Return={metrics['total_return_pct']}%")

            # Permutation test
            perm_p = permutation_test(returns, metrics['sharpe'])
            gates = five_gate(metrics, perm_p)

            variant_results[pair_name] = {
                'metrics': metrics,
                'perm_p': round(perm_p, 4),
                'five_gate': gates,
            }

            print(f"    perm_p={perm_p:.4f}, gates_pass={gates['pass_all']}")

        results[vname] = variant_results

    # Variant C: Multi-Pair Portfolio (runs all pairs together)
    print(f"\n{'='*60}")
    print(f"Running Variant C_multi_pair_portfolio")
    print(f"{'='*60}")

    returns_c, trades_c = variant_c_multipair(close, PAIRS, regime)
    if len(returns_c) > 0:
        metrics_c = compute_metrics(returns_c, regime, trades_c)
        perm_p_c = permutation_test(returns_c, metrics_c['sharpe'])
        gates_c = five_gate(metrics_c, perm_p_c)

        results['C_multi_pair_portfolio'] = {
            'all_pairs': {
                'metrics': metrics_c,
                'perm_p': round(perm_p_c, 4),
                'five_gate': gates_c,
            }
        }
        print(f"  Sharpe={metrics_c['sharpe']}, Trades={metrics_c['total_trades']}, "
              f"Return={metrics_c['total_return_pct']}%")
        print(f"  perm_p={perm_p_c:.4f}, gates_pass={gates_c['pass_all']}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    summary = {}
    for vname, vdata in results.items():
        for pair_or_all, pdata in vdata.items():
            key = f"{vname}__{pair_or_all}"
            m = pdata['metrics']
            summary[key] = {
                'sharpe': m['sharpe'],
                'total_return_pct': m['total_return_pct'],
                'trades': m['total_trades'],
                'pass': pdata['five_gate']['pass_all'],
            }
            status = "PASS" if pdata['five_gate']['pass_all'] else "FAIL"
            print(f"  {key}: Sharpe={m['sharpe']}, Ret={m['total_return_pct']}%, "
                  f"Trades={m['total_trades']}, [{status}]")

    # Build final output
    output = {
        'metadata': {
            'strategy': 'Pairs Trading / Statistical Arbitrage',
            'run_date': datetime.now().isoformat(),
            'oot_period': f'{OOT_START} to {END}',
            'starting_capital': CAPITAL,
            'commission': COMMISSION,
            'n_permutations': N_PERMS,
            'pairs': [f"{a}/{b}" for a, b in PAIRS],
            'robinhood_constraints': 'No shorting - long leg only',
        },
        'variants': results,
        'summary': summary,
    }

    out_path = '/home/jupiter/Lvl3Quant/data/pairs_stat_arb_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
