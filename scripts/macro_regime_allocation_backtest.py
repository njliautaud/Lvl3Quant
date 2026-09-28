#!/usr/bin/env python3
"""
Macro Regime Allocation Backtest
Tests quality stock basket with macro overlays vs always-invested baseline.

Variants:
  A: Always 100% quality basket (baseline)
  B: 100% quality unless VIX > 30 → 50% cash
  C: 100% quality unless SPY < 200-SMA AND VIX > 25 → 100% cash
  D: Tiered VIX allocation
  E: Multi-condition (SPY<200SMA, VIX>25, yield curve inverted) → 50% cash if 2/3
  F: Dual momentum vs TLT
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
QUALITY_TICKERS = ['AAPL', 'MSFT', 'AVGO', 'JPM', 'JNJ', 'PG', 'KO', 'PEP']
START_DATE = '2021-06-01'  # extra lookback for SMAs / momentum
OOT_START = '2022-01-03'
OOT_END = '2026-07-31'
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
N_PERMUTATIONS = 1000

# Known yield-curve inversion periods (10Y-2Y < 0)
# Source: FRED T10Y2Y — major inversion episodes
INVERSION_PERIODS = [
    ('2022-07-05', '2024-12-20'),  # prolonged 2022-2024 inversion
]


def download_data():
    """Download all required price data."""
    all_tickers = QUALITY_TICKERS + ['SPY', 'TLT', 'SHY', '^VIX', '^TNX']
    print("Downloading data...")
    data = yf.download(all_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data

    # Also try to get 10Y-2Y spread from ^TNX (10Y yield)
    # We'll use hardcoded inversion periods as primary

    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


def build_yield_curve_inverted_series(dates):
    """Return boolean series: True if yield curve inverted on that date."""
    inverted = pd.Series(False, index=dates)
    for start, end in INVERSION_PERIODS:
        mask = (dates >= start) & (dates <= end)
        inverted[mask] = True
    return inverted


def compute_quality_basket_returns(close_df, oot_dates):
    """Compute equal-weight daily returns for the quality basket."""
    quality_close = close_df[QUALITY_TICKERS].loc[oot_dates]
    quality_returns = quality_close.pct_change()
    # Equal weight
    basket_return = quality_returns.mean(axis=1)
    return basket_return


def apply_slippage(turnover_pct):
    """Compute round-trip slippage cost given turnover percentage."""
    return turnover_pct * SLIPPAGE_PCT * 2  # each way


def run_variant_a(basket_returns, oot_dates, close_df):
    """Always 100% quality basket, monthly rebalance."""
    capital = STARTING_CAPITAL
    equity_curve = [capital]
    rebalance_dates = []
    daily_returns = []

    current_month = oot_dates[0].month

    for i in range(1, len(oot_dates)):
        dt = oot_dates[i]
        ret = basket_returns.iloc[i]

        if pd.isna(ret):
            daily_returns.append(0.0)
            equity_curve.append(capital)
            continue

        # Monthly rebalance check
        if dt.month != current_month:
            # Rebalance: assume ~12.5% turnover for equal-weight rebalance (drift correction)
            slippage_cost = apply_slippage(0.125)
            capital *= (1 - slippage_cost)
            rebalance_dates.append(dt)
            current_month = dt.month

        daily_ret = ret
        capital *= (1 + daily_ret)
        daily_returns.append(daily_ret)
        equity_curve.append(capital)

    return np.array(equity_curve), np.array(daily_returns), rebalance_dates


def run_variant_b(basket_returns, oot_dates, close_df):
    """100% quality unless VIX > 30 → 50% cash."""
    vix = close_df['^VIX'].reindex(oot_dates).ffill()
    capital = STARTING_CAPITAL
    equity_curve = [capital]
    rebalance_dates = []
    daily_returns = []
    current_month = oot_dates[0].month
    prev_alloc = 1.0
    regime_signals = []

    for i in range(1, len(oot_dates)):
        dt = oot_dates[i]
        ret = basket_returns.iloc[i]
        if pd.isna(ret):
            daily_returns.append(0.0)
            equity_curve.append(capital)
            regime_signals.append(0)
            continue

        v = vix.iloc[i] if i < len(vix) else vix.iloc[-1]
        alloc = 0.5 if (not pd.isna(v) and v > 30) else 1.0
        regime_signals.append(1 if alloc < 1.0 else 0)

        # Rebalance on allocation change or monthly
        if dt.month != current_month or alloc != prev_alloc:
            turnover = abs(alloc - prev_alloc) + (0.125 * alloc if dt.month != current_month else 0)
            slippage_cost = apply_slippage(turnover)
            capital *= (1 - slippage_cost)
            rebalance_dates.append(dt)
            if dt.month != current_month:
                current_month = dt.month
            prev_alloc = alloc

        daily_ret = alloc * ret
        capital *= (1 + daily_ret)
        daily_returns.append(daily_ret)
        equity_curve.append(capital)

    return np.array(equity_curve), np.array(daily_returns), rebalance_dates, regime_signals


def run_variant_c(basket_returns, oot_dates, close_df):
    """100% quality unless SPY < 200-SMA AND VIX > 25 → 100% cash. Re-enter when SPY > 200-SMA."""
    spy_close = close_df['SPY']
    spy_sma200 = spy_close.rolling(200).mean()
    vix = close_df['^VIX'].reindex(oot_dates).ffill()
    spy_sma_oot = spy_sma200.reindex(oot_dates).ffill()
    spy_oot = spy_close.reindex(oot_dates).ffill()

    capital = STARTING_CAPITAL
    equity_curve = [capital]
    rebalance_dates = []
    daily_returns = []
    current_month = oot_dates[0].month
    in_cash = False
    prev_alloc = 1.0
    regime_signals = []

    for i in range(1, len(oot_dates)):
        dt = oot_dates[i]
        ret = basket_returns.iloc[i]
        if pd.isna(ret):
            daily_returns.append(0.0)
            equity_curve.append(capital)
            regime_signals.append(0)
            continue

        v = vix.iloc[i] if i < len(vix) else 20
        s = spy_oot.iloc[i] if i < len(spy_oot) else 400
        sma = spy_sma_oot.iloc[i] if i < len(spy_sma_oot) else 400

        if pd.isna(v): v = 20
        if pd.isna(s): s = 400
        if pd.isna(sma): sma = 400

        # Entry/exit logic
        if not in_cash and s < sma and v > 25:
            in_cash = True
        elif in_cash and s > sma:
            in_cash = False

        alloc = 0.0 if in_cash else 1.0
        regime_signals.append(1 if in_cash else 0)

        if alloc != prev_alloc or (not in_cash and dt.month != current_month):
            turnover = abs(alloc - prev_alloc) + (0.125 * alloc if dt.month != current_month else 0)
            slippage_cost = apply_slippage(turnover)
            capital *= (1 - slippage_cost)
            rebalance_dates.append(dt)
            prev_alloc = alloc

        if dt.month != current_month:
            current_month = dt.month

        daily_ret = alloc * ret
        capital *= (1 + daily_ret)
        daily_returns.append(daily_ret)
        equity_curve.append(capital)

    return np.array(equity_curve), np.array(daily_returns), rebalance_dates, regime_signals


def run_variant_d(basket_returns, oot_dates, close_df):
    """Tiered VIX: <20→100%, 20-30→70%, >30→30%."""
    vix = close_df['^VIX'].reindex(oot_dates).ffill()

    capital = STARTING_CAPITAL
    equity_curve = [capital]
    rebalance_dates = []
    daily_returns = []
    current_month = oot_dates[0].month
    prev_alloc = 1.0
    regime_signals = []

    for i in range(1, len(oot_dates)):
        dt = oot_dates[i]
        ret = basket_returns.iloc[i]
        if pd.isna(ret):
            daily_returns.append(0.0)
            equity_curve.append(capital)
            regime_signals.append(0)
            continue

        v = vix.iloc[i] if i < len(vix) else 20
        if pd.isna(v): v = 20

        if v < 20:
            alloc = 1.0
        elif v <= 30:
            alloc = 0.7
        else:
            alloc = 0.3

        regime_signals.append(1 if alloc < 1.0 else 0)

        if alloc != prev_alloc or dt.month != current_month:
            turnover = abs(alloc - prev_alloc) + (0.125 * alloc if dt.month != current_month else 0)
            slippage_cost = apply_slippage(turnover)
            capital *= (1 - slippage_cost)
            rebalance_dates.append(dt)
            if dt.month != current_month:
                current_month = dt.month
            prev_alloc = alloc

        daily_ret = alloc * ret
        capital *= (1 + daily_ret)
        daily_returns.append(daily_ret)
        equity_curve.append(capital)

    return np.array(equity_curve), np.array(daily_returns), rebalance_dates, regime_signals


def run_variant_e(basket_returns, oot_dates, close_df):
    """Multi-condition: 2 of 3 → 50% cash. (SPY<200SMA, VIX>25, yield curve inverted)."""
    spy_close = close_df['SPY']
    spy_sma200 = spy_close.rolling(200).mean()
    vix = close_df['^VIX'].reindex(oot_dates).ffill()
    spy_sma_oot = spy_sma200.reindex(oot_dates).ffill()
    spy_oot = spy_close.reindex(oot_dates).ffill()
    yc_inverted = build_yield_curve_inverted_series(oot_dates)

    capital = STARTING_CAPITAL
    equity_curve = [capital]
    rebalance_dates = []
    daily_returns = []
    current_month = oot_dates[0].month
    prev_alloc = 1.0
    regime_signals = []

    for i in range(1, len(oot_dates)):
        dt = oot_dates[i]
        ret = basket_returns.iloc[i]
        if pd.isna(ret):
            daily_returns.append(0.0)
            equity_curve.append(capital)
            regime_signals.append(0)
            continue

        v = vix.iloc[i] if i < len(vix) else 20
        s = spy_oot.iloc[i] if i < len(spy_oot) else 400
        sma = spy_sma_oot.iloc[i] if i < len(spy_sma_oot) else 400
        if pd.isna(v): v = 20
        if pd.isna(s): s = 400
        if pd.isna(sma): sma = 400

        conditions_met = sum([
            s < sma,
            v > 25,
            yc_inverted.iloc[i] if i < len(yc_inverted) else False
        ])

        alloc = 0.5 if conditions_met >= 2 else 1.0
        regime_signals.append(1 if alloc < 1.0 else 0)

        if alloc != prev_alloc or dt.month != current_month:
            turnover = abs(alloc - prev_alloc) + (0.125 * alloc if dt.month != current_month else 0)
            slippage_cost = apply_slippage(turnover)
            capital *= (1 - slippage_cost)
            rebalance_dates.append(dt)
            if dt.month != current_month:
                current_month = dt.month
            prev_alloc = alloc

        daily_ret = alloc * ret
        capital *= (1 + daily_ret)
        daily_returns.append(daily_ret)
        equity_curve.append(capital)

    return np.array(equity_curve), np.array(daily_returns), rebalance_dates, regime_signals


def run_variant_f(basket_returns, oot_dates, close_df):
    """Dual momentum: hold quality if 3M ret > TLT 3M ret AND > 0. Else hold TLT. Monthly rebalance."""
    tlt_close = close_df['TLT'].reindex(oot_dates).ffill()
    tlt_returns = tlt_close.pct_change()

    # Build quality basket price series for momentum calc
    quality_close = close_df[QUALITY_TICKERS].reindex(close_df.index)
    quality_price = quality_close.mean(axis=1)  # equal-weight proxy
    quality_price_oot = quality_price.reindex(oot_dates).ffill()
    tlt_price_full = close_df['TLT'].reindex(close_df.index).ffill()

    capital = STARTING_CAPITAL
    equity_curve = [capital]
    rebalance_dates = []
    daily_returns = []
    current_month = oot_dates[0].month
    holding = 'quality'  # start in quality
    regime_signals = []

    for i in range(1, len(oot_dates)):
        dt = oot_dates[i]
        ret_q = basket_returns.iloc[i]
        ret_t = tlt_returns.iloc[i] if i < len(tlt_returns) else 0
        if pd.isna(ret_q): ret_q = 0
        if pd.isna(ret_t): ret_t = 0

        # Monthly rebalance decision
        if dt.month != current_month:
            current_month = dt.month

            # Compute 3-month (63 trading days) returns
            lookback = 63
            idx_in_full = close_df.index.get_indexer([dt], method='ffill')[0]
            if idx_in_full >= lookback:
                q_start = quality_price.iloc[idx_in_full - lookback]
                q_now = quality_price.iloc[idx_in_full]
                t_start = tlt_price_full.iloc[idx_in_full - lookback]
                t_now = tlt_price_full.iloc[idx_in_full]

                if q_start > 0 and t_start > 0:
                    q_mom = (q_now / q_start) - 1
                    t_mom = (t_now / t_start) - 1
                else:
                    q_mom = 0
                    t_mom = 0

                new_holding = 'quality' if (q_mom > t_mom and q_mom > 0) else 'tlt'
            else:
                new_holding = 'quality'

            if new_holding != holding:
                # Full turnover on switch
                slippage_cost = apply_slippage(1.0)
                capital *= (1 - slippage_cost)
            else:
                # Normal monthly rebalance
                slippage_cost = apply_slippage(0.125)
                capital *= (1 - slippage_cost)

            rebalance_dates.append(dt)
            holding = new_holding

        regime_signals.append(1 if holding == 'tlt' else 0)

        if holding == 'quality':
            daily_ret = ret_q
        else:
            daily_ret = ret_t

        capital *= (1 + daily_ret)
        daily_returns.append(daily_ret)
        equity_curve.append(capital)

    return np.array(equity_curve), np.array(daily_returns), rebalance_dates, regime_signals


def compute_metrics(daily_returns, equity_curve, name):
    """Compute standard performance metrics."""
    dr = np.array(daily_returns)
    dr = dr[~np.isnan(dr)]

    n_days = len(dr)
    if n_days < 2:
        return {}

    total_return = (equity_curve[-1] / equity_curve[0]) - 1
    years = n_days / 252
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    ann_vol = np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 1e-6
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_dd = np.min(drawdown)

    return {
        'name': name,
        'total_return_pct': round(total_return * 100, 2),
        'annualized_return_pct': round(ann_return * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'volatility_pct': round(ann_vol * 100, 2),
        'final_value': round(equity_curve[-1], 2),
        'n_trading_days': n_days,
    }


def regime_stratified_performance(daily_returns, oot_dates, close_df):
    """Split performance into bull (SPY > 200-SMA) and bear (SPY < 200-SMA) regimes."""
    spy_close = close_df['SPY']
    spy_sma200 = spy_close.rolling(200).mean()
    spy_sma_oot = spy_sma200.reindex(oot_dates).ffill()
    spy_oot = spy_close.reindex(oot_dates).ffill()

    dr = np.array(daily_returns)
    # Align — daily_returns has len(oot_dates)-1 entries
    dates_for_returns = oot_dates[1:]

    bull_rets = []
    bear_rets = []

    for i, dt in enumerate(dates_for_returns):
        if i >= len(dr):
            break
        s = spy_oot.loc[dt] if dt in spy_oot.index else np.nan
        sma = spy_sma_oot.loc[dt] if dt in spy_sma_oot.index else np.nan

        if pd.isna(s) or pd.isna(sma):
            continue

        if s >= sma:
            bull_rets.append(dr[i])
        else:
            bear_rets.append(dr[i])

    bull_rets = np.array(bull_rets) if bull_rets else np.array([0])
    bear_rets = np.array(bear_rets) if bear_rets else np.array([0])

    def sharpe_from_rets(r):
        if len(r) < 2 or np.std(r) == 0:
            return 0
        return (np.mean(r) * 252) / (np.std(r) * np.sqrt(252))

    bull_sharpe = sharpe_from_rets(bull_rets)
    bear_sharpe = sharpe_from_rets(bear_rets)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'bull_days': len(bull_rets),
        'bear_days': len(bear_rets),
        'regime_gap': round(regime_gap, 3),
    }


def permutation_test(daily_returns, regime_signals, n_perms=1000):
    """
    Permutation test: shuffle regime signals to check if timing adds value.
    Returns p-value (fraction of permutations with better Sharpe than actual).
    """
    dr = np.array(daily_returns)
    signals = np.array(regime_signals)

    if len(dr) != len(signals):
        min_len = min(len(dr), len(signals))
        dr = dr[:min_len]
        signals = signals[:min_len]

    # Actual strategy returns: when signal=1, reduce exposure
    # For comparison, compute actual Sharpe
    actual_sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252) if np.std(dr) > 0 else 0

    better_count = 0
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        shuffled = rng.permutation(signals)
        # Reconstruct returns with shuffled signals
        # We need the unscaled basket returns... approximate by adjusting
        # When signal=1, the actual return was already reduced.
        # We need original basket returns to do this properly.
        # Instead: test if the actual Sharpe is better than random timing
        perm_rets = dr.copy()
        # Swap the signal days randomly
        rng.shuffle(perm_rets)
        perm_sharpe = np.mean(perm_rets) / np.std(perm_rets) * np.sqrt(252) if np.std(perm_rets) > 0 else 0
        if perm_sharpe >= actual_sharpe:
            better_count += 1

    return better_count / n_perms


def permutation_test_proper(basket_returns_arr, regime_signals, n_perms=1000):
    """
    Proper permutation test: uses raw basket returns + shuffled regime signals
    to see if timing adds value vs random timing.
    """
    basket_rets = np.array(basket_returns_arr)
    signals = np.array(regime_signals)

    min_len = min(len(basket_rets), len(signals))
    basket_rets = basket_rets[:min_len]
    signals = signals[:min_len]

    def compute_sharpe_with_signals(rets, sigs):
        """Apply signals: signal=1 means reduced exposure (use 0.5 or 0 depending on variant)."""
        # Generic: signal=1 → 50% allocation, signal=0 → 100%
        allocs = np.where(sigs == 1, 0.5, 1.0)
        adj_rets = rets * allocs
        if np.std(adj_rets) == 0:
            return 0
        return np.mean(adj_rets) / np.std(adj_rets) * np.sqrt(252)

    actual_sharpe = compute_sharpe_with_signals(basket_rets, signals)

    better_count = 0
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        shuffled = rng.permutation(signals)
        perm_sharpe = compute_sharpe_with_signals(basket_rets, shuffled)
        if perm_sharpe >= actual_sharpe:
            better_count += 1

    return round(better_count / n_perms, 4), round(actual_sharpe, 4)


def five_gate_validation(metrics, perm_p, regime_gap, n_rebalances):
    """Apply 5-gate validation."""
    gates = {
        'sharpe_gt_0.5': metrics['sharpe'] > 0.5,
        'perm_test_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'maxdd_gt_neg50': metrics['max_drawdown_pct'] > -50,
        'rebalance_events_gte_20': n_rebalances >= 20,
    }
    gates['all_passed'] = all(gates.values())
    return gates


def main():
    close_df = download_data()

    # Filter to OOT period
    oot_mask = (close_df.index >= OOT_START) & (close_df.index <= OOT_END)
    oot_dates = close_df.index[oot_mask]
    print(f"OOT period: {oot_dates[0].date()} to {oot_dates[-1].date()}, {len(oot_dates)} trading days")

    basket_returns = compute_quality_basket_returns(close_df, oot_dates)
    basket_returns_arr = basket_returns.iloc[1:].values  # skip first NaN

    # ── Run all variants ────────────────────────────────────────────────────
    results = {}

    # Variant A — baseline
    print("\nRunning Variant A (always invested)...")
    eq_a, dr_a, reb_a = run_variant_a(basket_returns, oot_dates, close_df)
    m_a = compute_metrics(dr_a, eq_a, 'A: Always Invested')
    rs_a = regime_stratified_performance(dr_a, oot_dates, close_df)
    m_a.update(rs_a)
    m_a['n_rebalances'] = len(reb_a)
    m_a['perm_p'] = 'N/A (baseline)'
    m_a['gates'] = 'N/A (baseline)'
    results['A'] = m_a

    # Variants B-F
    variant_funcs = {
        'B': ('B: VIX>30 → 50% cash', run_variant_b),
        'C': ('C: SPY<200SMA & VIX>25 → cash', run_variant_c),
        'D': ('D: Tiered VIX allocation', run_variant_d),
        'E': ('E: Multi-condition 2/3 → 50% cash', run_variant_e),
        'F': ('F: Dual momentum vs TLT', run_variant_f),
    }

    for key, (name, func) in variant_funcs.items():
        print(f"Running Variant {key}...")
        eq, dr, reb, signals = func(basket_returns, oot_dates, close_df)
        m = compute_metrics(dr, eq, name)
        rs = regime_stratified_performance(dr, oot_dates, close_df)
        m.update(rs)
        m['n_rebalances'] = len(reb)

        # Permutation test
        perm_p, actual_sharpe = permutation_test_proper(basket_returns_arr, signals, N_PERMUTATIONS)
        m['perm_p'] = perm_p
        m['perm_actual_sharpe'] = actual_sharpe

        # Signal stats
        m['pct_time_reduced'] = round(np.mean(signals) * 100, 1)

        # 5-gate validation
        gates = five_gate_validation(m, perm_p, rs['regime_gap'], len(reb))
        m['gates'] = gates

        # Comparison to baseline A
        m['vs_baseline_sharpe_diff'] = round(m['sharpe'] - results['A']['sharpe'], 3)
        m['vs_baseline_return_diff_pct'] = round(m['total_return_pct'] - results['A']['total_return_pct'], 2)
        m['vs_baseline_maxdd_diff_pct'] = round(m['max_drawdown_pct'] - results['A']['max_drawdown_pct'], 2)

        results[key] = m

    # ── Print Summary Table ─────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("MACRO REGIME ALLOCATION BACKTEST — OOT Jan 2022 – Jul 2026")
    print(f"Starting Capital: ${STARTING_CAPITAL:.0f}")
    print("=" * 120)

    header = f"{'Variant':<40} {'Return%':>8} {'Ann%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Final$':>8} {'Rebal':>6} {'PermP':>7} {'RGap':>6}"
    print(header)
    print("-" * 120)

    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        m = results[key]
        perm_str = f"{m['perm_p']:.3f}" if isinstance(m['perm_p'], float) else m['perm_p']
        rg = m.get('regime_gap', 'N/A')
        rg_str = f"{rg:.3f}" if isinstance(rg, float) else str(rg)
        print(f"{m['name']:<40} {m['total_return_pct']:>7.1f}% {m['annualized_return_pct']:>6.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_drawdown_pct']:>7.1f}% ${m['final_value']:>7.0f} {m['n_rebalances']:>6} {perm_str:>7} {rg_str:>6}")

    print("\n" + "-" * 120)
    print("COMPARISON TO BASELINE (Variant A):")
    print(f"{'Variant':<40} {'Sharpe Diff':>12} {'Return Diff':>12} {'MaxDD Diff':>12} {'% Time Reduced':>15}")
    print("-" * 120)
    for key in ['B', 'C', 'D', 'E', 'F']:
        m = results[key]
        print(f"{m['name']:<40} {m['vs_baseline_sharpe_diff']:>+11.3f} {m['vs_baseline_return_diff_pct']:>+11.1f}% {m['vs_baseline_maxdd_diff_pct']:>+11.1f}% {m['pct_time_reduced']:>14.1f}%")

    print("\n" + "-" * 120)
    print("5-GATE VALIDATION:")
    print(f"{'Variant':<40} {'Sharpe>0.5':>11} {'PermP<0.05':>11} {'RGap<0.5':>11} {'DD>-50%':>11} {'Rebal≥20':>11} {'PASS':>6}")
    print("-" * 120)
    for key in ['B', 'C', 'D', 'E', 'F']:
        m = results[key]
        g = m['gates']
        pass_str = "YES" if g['all_passed'] else "NO"
        print(f"{m['name']:<40} {'PASS' if g['sharpe_gt_0.5'] else 'FAIL':>11} {'PASS' if g['perm_test_p_lt_0.05'] else 'FAIL':>11} {'PASS' if g['regime_gap_lt_0.5'] else 'FAIL':>11} {'PASS' if g['maxdd_gt_neg50'] else 'FAIL':>11} {'PASS' if g['rebalance_events_gte_20'] else 'FAIL':>11} {pass_str:>6}")

    print("\n" + "-" * 120)
    print("REGIME STRATIFICATION (Bull = SPY > 200-SMA, Bear = SPY < 200-SMA):")
    print(f"{'Variant':<40} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Bull Days':>10} {'Bear Days':>10} {'Regime Gap':>11}")
    print("-" * 120)
    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        m = results[key]
        print(f"{m['name']:<40} {m['bull_sharpe']:>12.3f} {m['bear_sharpe']:>12.3f} {m['bull_days']:>10} {m['bear_days']:>10} {m['regime_gap']:>11.3f}")

    # ── Key Finding ─────────────────────────────────────────────────────────
    best_overlay = max(['B', 'C', 'D', 'E', 'F'], key=lambda k: results[k]['sharpe'])
    print("\n" + "=" * 120)
    print("KEY FINDING:")
    baseline_sharpe = results['A']['sharpe']
    best_sharpe = results[best_overlay]['sharpe']
    if best_sharpe > baseline_sharpe:
        print(f"  Best overlay: Variant {best_overlay} ({results[best_overlay]['name']}) with Sharpe {best_sharpe:.3f} vs baseline {baseline_sharpe:.3f}")
        print(f"  Timing DOES add value: +{best_sharpe - baseline_sharpe:.3f} Sharpe improvement")
    else:
        print(f"  NO overlay beats the always-invested baseline (Sharpe {baseline_sharpe:.3f})")
        print(f"  Best overlay attempt: {best_overlay} with Sharpe {best_sharpe:.3f}")
        print("  CONCLUSION: Quality stocks are inherently regime-neutral. Timing adds no value.")

    any_passed = any(results[k]['gates']['all_passed'] for k in ['B', 'C', 'D', 'E', 'F'] if isinstance(results[k]['gates'], dict))
    if not any_passed:
        print("  No variant passed all 5 gates.")
    print("=" * 120)

    # ── Save Results ────────────────────────────────────────────────────────
    output_path = Path('/home/jupiter/Lvl3Quant/data/macro_regime_allocation_results.json')

    # Convert for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(x) for x in obj]
        return obj

    output = {
        'metadata': {
            'oot_start': OOT_START,
            'oot_end': str(oot_dates[-1].date()),
            'starting_capital': STARTING_CAPITAL,
            'slippage_pct': SLIPPAGE_PCT,
            'quality_tickers': QUALITY_TICKERS,
            'n_permutations': N_PERMUTATIONS,
            'run_timestamp': datetime.now().isoformat(),
        },
        'results': make_serializable(results),
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
