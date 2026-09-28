#!/usr/bin/env python3
"""
VIX Term Structure / Volatility Crush Timing Strategy — v1
============================================================
Tests 6 VIX-based timing signals on SPY across 5 hold periods.
Equity sim first, then BS-priced call debit spread sim.

ALL adversarial checks built INLINE per HC #705:
  - Permutation test (random DATE entry, NOT return shuffling)
  - Regime stratification (SPY green/red/flat periods)
  - Sub-period consistency (pre-2018, 2018-2022, 2022+)
  - Outlier removal (winsorize top/bottom 5%)
  - Drawdown analysis
  - Signal frequency reporting

Entry signals:
  1. VIX_spike_10pct:     VIX rises 10%+ in a single day
  2. VIX_above_20:        VIX crosses above 20 from below
  3. VIX_mean_rev_25:     VIX > 25
  4. VIX_mean_rev_20:     VIX > 20
  5. VIX_drop_after_spike: VIX was > 25 within 5 days, now < 22
  6. VIX_contango_proxy:  VIX/VIX_20MA > 1.15 (fear spike)
"""

import sys, json, warnings, os, time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "vix_timing_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIG
# =============================================================================

STARTING_CAPITAL       = 10_000
RISK_PER_TRADE         = 200       # $200 per trade for small account
MAX_CONCURRENT         = 2
EQUITY_SLIPPAGE_PCT    = 0.001     # 10 bps
RISK_FREE_RATE         = 0.045
N_PERMUTATIONS         = 200       # robust permutation test
SPREAD_WIDTH_PCT       = 0.02      # ATM / ATM+2% call debit spread

HOLD_PERIODS = [1, 3, 5, 10, 20]

# =============================================================================
# DATA DOWNLOAD
# =============================================================================

def download_data():
    """Download SPY + ^VIX daily data 2010-2026."""
    import yfinance as yf

    cache_file = OUTPUT / "data_cache.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        if len(df) > 3000:  # sanity check
            print(f"  Loaded cached data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
            return df

    print("  Downloading SPY + ^VIX from yfinance...")
    spy = yf.download("SPY", start="2010-01-01", end="2026-07-15", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start="2010-01-01", end="2026-07-15", auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    df = pd.DataFrame(index=spy.index)
    df['spy_open'] = spy['Open']
    df['spy_close'] = spy['Close']
    df['spy_high'] = spy['High']
    df['spy_low'] = spy['Low']
    df['vix_close'] = vix['Close'].reindex(spy.index)
    df['vix_open'] = vix['Open'].reindex(spy.index)

    df = df.dropna()

    # Derived features
    df['vix_pct_change'] = df['vix_close'].pct_change()
    df['vix_20ma'] = df['vix_close'].rolling(20).mean()
    df['vix_ratio_to_ma'] = df['vix_close'] / df['vix_20ma']
    df['spy_ret_1d'] = df['spy_close'].pct_change()

    # VIX was > 25 within last 5 days
    df['vix_was_above_25_5d'] = df['vix_close'].rolling(5).max() > 25

    # VIX crossed above 20 (was below yesterday, above today)
    df['vix_prev'] = df['vix_close'].shift(1)

    df = df.dropna()

    df.to_parquet(cache_file)
    print(f"  Data: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    return df


# =============================================================================
# SIGNAL GENERATORS
# =============================================================================

def generate_signals(df):
    """Generate all 6 signal series. Returns dict of signal_name -> boolean Series."""
    signals = {}

    # 1. VIX_spike_10pct: VIX rises 10%+ in a single day
    signals['VIX_spike_10pct'] = df['vix_pct_change'] >= 0.10

    # 2. VIX_above_20: VIX crosses above 20 from below
    signals['VIX_above_20'] = (df['vix_close'] >= 20) & (df['vix_prev'] < 20)

    # 3. VIX_mean_rev_25: VIX > 25
    signals['VIX_mean_rev_25'] = df['vix_close'] > 25

    # 4. VIX_mean_rev_20: VIX > 20
    signals['VIX_mean_rev_20'] = df['vix_close'] > 20

    # 5. VIX_drop_after_spike: VIX was > 25 within 5d but now < 22
    signals['VIX_drop_after_spike'] = (df['vix_was_above_25_5d']) & (df['vix_close'] < 22)

    # 6. VIX_contango_proxy: VIX/VIX_20MA > 1.15 (fear spike relative to norm)
    signals['VIX_contango_proxy'] = df['vix_ratio_to_ma'] > 1.15

    return signals


# =============================================================================
# BLACK-SCHOLES PRICING
# =============================================================================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def price_debit_spread(spy_price, vix_level, hold_days, spread_width_pct=SPREAD_WIDTH_PCT):
    """
    Price ATM / ATM+spread_width call debit spread using BS.
    Returns (debit_cost, max_profit, breakeven_pct).
    """
    K1 = spy_price                              # ATM long call
    K2 = spy_price * (1 + spread_width_pct)     # OTM short call
    T = hold_days / 252.0
    sigma = vix_level / 100.0  # VIX is annualized vol in %

    c1 = bs_call_price(spy_price, K1, T, RISK_FREE_RATE, sigma)
    c2 = bs_call_price(spy_price, K2, T, RISK_FREE_RATE, sigma)

    debit = c1 - c2
    max_profit = (K2 - K1) - debit

    if debit <= 0:
        debit = 0.01  # floor

    return debit, max_profit


def spread_pnl_at_expiry(spy_entry, spy_exit, vix_at_entry, hold_days):
    """
    Calculate P&L of the debit spread at expiry.
    Returns (pnl_pct, debit_cost).
    """
    debit, max_profit = price_debit_spread(spy_entry, vix_at_entry, hold_days)

    K1 = spy_entry
    K2 = spy_entry * (1 + SPREAD_WIDTH_PCT)

    # Intrinsic value at expiry
    long_intrinsic = max(spy_exit - K1, 0)
    short_intrinsic = max(spy_exit - K2, 0)
    spread_value = long_intrinsic - short_intrinsic

    pnl = spread_value - debit
    pnl_pct = pnl / debit if debit > 0 else 0

    return pnl_pct, debit


# =============================================================================
# EQUITY SIMULATION
# =============================================================================

def run_equity_sim(df, signal_dates, hold_days):
    """
    Simple equity sim: buy SPY at next open after signal, sell after hold_days.
    Returns list of trade dicts.
    """
    trades = []

    all_dates = df.index.tolist()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for sig_date in signal_dates:
        idx = date_to_idx.get(sig_date)
        if idx is None or idx + 1 >= len(all_dates):
            continue

        entry_idx = idx + 1  # buy next open
        exit_idx = entry_idx + hold_days

        if exit_idx >= len(all_dates):
            continue

        entry_date = all_dates[entry_idx]
        exit_date = all_dates[exit_idx]

        entry_price = df.loc[entry_date, 'spy_open']
        exit_price = df.loc[exit_date, 'spy_close']

        # Apply slippage
        entry_price *= (1 + EQUITY_SLIPPAGE_PCT)
        exit_price *= (1 - EQUITY_SLIPPAGE_PCT)

        ret = (exit_price - entry_price) / entry_price

        trades.append({
            'signal_date': sig_date,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'return': ret,
            'vix_at_signal': df.loc[sig_date, 'vix_close'],
        })

    return trades


def run_spread_sim(df, signal_dates, hold_days):
    """
    BS-priced call debit spread sim.
    Returns list of trade dicts with spread P&L.
    """
    trades = []
    all_dates = df.index.tolist()
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for sig_date in signal_dates:
        idx = date_to_idx.get(sig_date)
        if idx is None or idx + 1 >= len(all_dates):
            continue

        entry_idx = idx + 1
        exit_idx = entry_idx + hold_days

        if exit_idx >= len(all_dates):
            continue

        entry_date = all_dates[entry_idx]
        exit_date = all_dates[exit_idx]

        spy_entry = df.loc[entry_date, 'spy_open']
        spy_exit = df.loc[exit_date, 'spy_close']
        vix_at_entry = df.loc[sig_date, 'vix_close']

        pnl_pct, debit = spread_pnl_at_expiry(spy_entry, spy_exit, vix_at_entry, hold_days)

        # Position size
        n_spreads = max(1, int(RISK_PER_TRADE / (debit * 100)))
        dollar_pnl = pnl_pct * debit * 100 * n_spreads

        trades.append({
            'signal_date': sig_date,
            'entry_date': entry_date,
            'exit_date': exit_date,
            'spy_entry': spy_entry,
            'spy_exit': spy_exit,
            'vix_at_signal': vix_at_entry,
            'debit_per_spread': round(debit, 4),
            'n_spreads': n_spreads,
            'pnl_pct': pnl_pct,
            'dollar_pnl': dollar_pnl,
        })

    return trades


# =============================================================================
# ADVERSARIAL CHECKS (ALL INLINE PER HC #705)
# =============================================================================

def permutation_test_random_dates(df, observed_mean_ret, n_signals, hold_days, n_perms=N_PERMUTATIONS):
    """
    CRITICAL: Random DATE entry permutation test (NOT return shuffling).
    For each permutation, randomly select n_signals entry dates from the full
    date range, compute forward returns, compare mean to observed.
    """
    all_dates = df.index.tolist()
    max_idx = len(all_dates) - hold_days - 2  # leave room for hold period

    if max_idx < n_signals or n_signals < 3:
        return None, None

    eligible_indices = list(range(20, max_idx))  # skip first 20 for warmup

    perm_means = []
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        random_indices = rng.choice(eligible_indices, size=n_signals, replace=False)
        rets = []
        for idx in random_indices:
            entry_idx = idx + 1
            exit_idx = entry_idx + hold_days
            if exit_idx >= len(all_dates):
                continue
            entry_p = df.iloc[entry_idx]['spy_open'] * (1 + EQUITY_SLIPPAGE_PCT)
            exit_p = df.iloc[exit_idx]['spy_close'] * (1 - EQUITY_SLIPPAGE_PCT)
            rets.append((exit_p - entry_p) / entry_p)
        if rets:
            perm_means.append(np.mean(rets))

    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= observed_mean_ret)

    return p_value, perm_means


def regime_stratification(trades_df, df):
    """
    Split trades by market regime at entry:
    - Green: SPY 20d return > +2%
    - Red: SPY 20d return < -2%
    - Flat: otherwise
    """
    df_temp = df.copy()
    df_temp['spy_20d_ret'] = df_temp['spy_close'].pct_change(20)

    results = {}
    for regime, condition in [
        ('green', df_temp['spy_20d_ret'] > 0.02),
        ('red', df_temp['spy_20d_ret'] < -0.02),
        ('flat', (df_temp['spy_20d_ret'] >= -0.02) & (df_temp['spy_20d_ret'] <= 0.02)),
    ]:
        regime_dates = set(df_temp[condition].index)
        regime_trades = trades_df[trades_df['signal_date'].isin(regime_dates)]

        if len(regime_trades) >= 3:
            rets = regime_trades['return'].values
            results[regime] = {
                'n': len(regime_trades),
                'mean_ret': float(np.mean(rets)),
                'win_rate': float(np.mean(rets > 0)),
                'sharpe': float(np.mean(rets) / np.std(rets) * np.sqrt(252 / max(1, len(rets)))) if np.std(rets) > 0 else 0,
            }
        else:
            results[regime] = {'n': len(regime_trades), 'mean_ret': None, 'win_rate': None, 'sharpe': None}

    return results


def sub_period_consistency(trades_df):
    """Check consistency across sub-periods."""
    results = {}
    for period_name, start, end in [
        ('2010-2017', '2010-01-01', '2017-12-31'),
        ('2018-2021', '2018-01-01', '2021-12-31'),
        ('2022-2026', '2022-01-01', '2026-12-31'),
    ]:
        mask = (trades_df['signal_date'] >= start) & (trades_df['signal_date'] <= end)
        period_trades = trades_df[mask]

        if len(period_trades) >= 3:
            rets = period_trades['return'].values
            results[period_name] = {
                'n': len(period_trades),
                'mean_ret': float(np.mean(rets)),
                'win_rate': float(np.mean(rets > 0)),
                'sharpe': float(np.mean(rets) / np.std(rets) * np.sqrt(252 / max(1, len(rets)))) if np.std(rets) > 0 else 0,
            }
        else:
            results[period_name] = {'n': len(period_trades), 'mean_ret': None, 'win_rate': None, 'sharpe': None}

    return results


def outlier_analysis(trades_df):
    """Winsorize top/bottom 5% and recompute metrics."""
    rets = trades_df['return'].values.copy()
    if len(rets) < 10:
        return None

    lo, hi = np.percentile(rets, [5, 95])
    winsorized = np.clip(rets, lo, hi)

    return {
        'original_mean': float(np.mean(rets)),
        'winsorized_mean': float(np.mean(winsorized)),
        'pct_change': float((np.mean(winsorized) - np.mean(rets)) / abs(np.mean(rets)) * 100) if np.mean(rets) != 0 else 0,
        'n_clipped': int(np.sum((rets < lo) | (rets > hi))),
        'original_wr': float(np.mean(rets > 0)),
        'winsorized_wr': float(np.mean(winsorized > 0)),
    }


def drawdown_analysis(trades_df):
    """Compute max drawdown and max consecutive losses."""
    rets = trades_df['return'].values
    if len(rets) < 2:
        return None

    # Equity curve
    equity = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(np.min(dd))

    # Max consecutive losses
    losses = (rets < 0).astype(int)
    max_consec = 0
    current = 0
    for l in losses:
        if l:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0

    return {
        'max_drawdown_pct': round(max_dd * 100, 2),
        'max_consecutive_losses': max_consec,
        'total_trades': len(rets),
    }


# =============================================================================
# MAIN BACKTEST
# =============================================================================

def compute_metrics(rets):
    """Compute standard metrics from array of returns."""
    if len(rets) < 2:
        return {}

    mean_r = np.mean(rets)
    std_r = np.std(rets)

    # Annualization factor (approximate)
    ann = np.sqrt(252 / max(1, len(rets)))

    sharpe = (mean_r / std_r) * ann if std_r > 0 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside) if len(downside) > 1 else std_r
    sortino = (mean_r / downside_std) * ann if downside_std > 0 else 0

    # Profit factor
    gains = rets[rets > 0]
    losses_abs = np.abs(rets[rets < 0])
    pf = float(np.sum(gains) / np.sum(losses_abs)) if np.sum(losses_abs) > 0 else float('inf')

    return {
        'n_trades': len(rets),
        'mean_return_pct': round(mean_r * 100, 3),
        'median_return_pct': round(float(np.median(rets)) * 100, 3),
        'win_rate': round(float(np.mean(rets > 0)) * 100, 1),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 3),
        'total_return_pct': round(float(np.prod(1 + rets) - 1) * 100, 2),
        'std_pct': round(std_r * 100, 3),
        'best_pct': round(float(np.max(rets)) * 100, 2),
        'worst_pct': round(float(np.min(rets)) * 100, 2),
    }


def main():
    print("=" * 80)
    print("VIX TERM STRUCTURE / VOLATILITY CRUSH TIMING — v1")
    print("=" * 80)

    # ---- DATA ----
    print("\n[1/4] LOADING DATA...")
    df = download_data()
    signals = generate_signals(df)

    # Report signal frequencies
    print("\n  Signal frequencies:")
    for name, sig in signals.items():
        n = sig.sum()
        freq_per_year = n / (len(df) / 252)
        print(f"    {name:30s}: {n:5d} signals ({freq_per_year:.1f}/year)")

    # ---- EQUITY SIM ----
    print("\n[2/4] EQUITY SIMULATION (SPY)...")

    all_results = {}

    for sig_name, sig_mask in signals.items():
        signal_dates = df.index[sig_mask].tolist()

        if len(signal_dates) < 5:
            print(f"  {sig_name}: SKIP (only {len(signal_dates)} signals)")
            continue

        all_results[sig_name] = {}

        for hold in HOLD_PERIODS:
            trades = run_equity_sim(df, signal_dates, hold)

            if len(trades) < 5:
                continue

            trades_df = pd.DataFrame(trades)
            trades_df['signal_date'] = pd.to_datetime(trades_df['signal_date'])
            rets = trades_df['return'].values

            metrics = compute_metrics(rets)

            # --- ADVERSARIAL CHECKS ---

            # 1. Permutation test (random date entry)
            p_val, perm_means = permutation_test_random_dates(
                df, np.mean(rets), len(signal_dates), hold
            )

            # 2. Regime stratification
            regimes = regime_stratification(trades_df, df)

            # 3. Sub-period consistency
            sub_periods = sub_period_consistency(trades_df)

            # 4. Outlier analysis
            outlier = outlier_analysis(trades_df)

            # 5. Drawdown analysis
            dd = drawdown_analysis(trades_df)

            # --- PASS/FAIL GATES ---
            gates = {}

            # Gate 1: Permutation p-value < 0.05
            if p_val is not None:
                gates['permutation_p<0.05'] = p_val < 0.05
                gates['permutation_p_value'] = round(p_val, 4)

            # Gate 2: Win rate > 50%
            gates['win_rate>50%'] = metrics.get('win_rate', 0) > 50

            # Gate 3: Profit factor > 1.0
            gates['profit_factor>1.0'] = metrics.get('profit_factor', 0) > 1.0

            # Gate 4: Works in at least 2 of 3 sub-periods
            n_profitable_periods = sum(
                1 for p in sub_periods.values()
                if p.get('mean_ret') is not None and p['mean_ret'] > 0
            )
            gates['profitable_in_2+_subperiods'] = n_profitable_periods >= 2
            gates['n_profitable_subperiods'] = n_profitable_periods

            # Gate 5: Not regime-dependent (works in both green and red)
            green_wr = regimes.get('green', {}).get('win_rate')
            red_wr = regimes.get('red', {}).get('win_rate')
            if green_wr is not None and red_wr is not None:
                gates['regime_robust'] = min(green_wr, red_wr) > 0.40
            else:
                gates['regime_robust'] = None  # insufficient data

            # Gate 6: Outlier-robust (winsorized mean still positive)
            if outlier:
                gates['outlier_robust'] = outlier['winsorized_mean'] > 0

            # Gate 7: Mean return > cost of trading (slippage)
            gates['exceeds_costs'] = metrics.get('mean_return_pct', 0) > 0.2  # > 20bps

            # Overall pass
            critical_gates = ['permutation_p<0.05', 'win_rate>50%', 'profit_factor>1.0',
                            'profitable_in_2+_subperiods', 'outlier_robust', 'exceeds_costs']
            passes = sum(1 for g in critical_gates if gates.get(g) == True)
            total = sum(1 for g in critical_gates if gates.get(g) is not None)
            gates['gates_passed'] = f"{passes}/{total}"
            gates['ALL_PASS'] = passes == total and total >= 4

            result = {
                'metrics': metrics,
                'gates': gates,
                'regimes': regimes,
                'sub_periods': sub_periods,
                'outlier': outlier,
                'drawdown': dd,
            }

            all_results[sig_name][f'hold_{hold}d'] = result

    # ---- SPREAD SIM ----
    print("\n[3/4] CALL DEBIT SPREAD SIMULATION...")

    spread_results = {}

    for sig_name, sig_mask in signals.items():
        signal_dates = df.index[sig_mask].tolist()

        if len(signal_dates) < 5:
            continue

        spread_results[sig_name] = {}

        for hold in HOLD_PERIODS:
            trades = run_spread_sim(df, signal_dates, hold)

            if len(trades) < 5:
                continue

            trades_df = pd.DataFrame(trades)
            pnl_pcts = trades_df['pnl_pct'].values
            dollar_pnls = trades_df['dollar_pnl'].values

            # Equity curve with $10K start
            cumulative = STARTING_CAPITAL + np.cumsum(dollar_pnls)
            peak = np.maximum.accumulate(cumulative)
            dd = (cumulative - peak) / peak
            max_dd = float(np.min(dd))

            total_pnl = float(np.sum(dollar_pnls))

            spread_metrics = {
                'n_trades': len(trades),
                'total_pnl': round(total_pnl, 2),
                'mean_pnl_per_trade': round(float(np.mean(dollar_pnls)), 2),
                'win_rate': round(float(np.mean(pnl_pcts > 0)) * 100, 1),
                'mean_return_pct': round(float(np.mean(pnl_pcts)) * 100, 2),
                'max_drawdown_pct': round(max_dd * 100, 2),
                'final_equity': round(float(cumulative[-1]), 2),
                'avg_debit': round(float(trades_df['debit_per_spread'].mean()), 4),
                'avg_n_spreads': round(float(trades_df['n_spreads'].mean()), 1),
                'profit_factor': round(
                    float(np.sum(dollar_pnls[dollar_pnls > 0]) / abs(np.sum(dollar_pnls[dollar_pnls < 0])))
                    if np.sum(dollar_pnls[dollar_pnls < 0]) != 0 else float('inf'), 3
                ),
            }

            spread_results[sig_name][f'hold_{hold}d'] = spread_metrics

    # ---- PRINT RESULTS ----
    print("\n[4/4] RESULTS")
    print("=" * 80)

    # Summary table — equity
    print("\n" + "=" * 80)
    print("EQUITY SIM SUMMARY (SPY)")
    print("=" * 80)

    header = f"{'Signal':<25s} {'Hold':>4s} {'N':>5s} {'Mean%':>7s} {'WR%':>5s} {'Sharpe':>7s} {'Sort':>7s} {'PF':>6s} {'Perm-p':>7s} {'Gates':>7s} {'PASS':>5s}"
    print(header)
    print("-" * len(header))

    passing_combos = []

    for sig_name in sorted(all_results.keys()):
        for hold_key in sorted(all_results[sig_name].keys()):
            r = all_results[sig_name][hold_key]
            m = r['metrics']
            g = r['gates']
            hold_str = hold_key.replace('hold_', '').replace('d', '')

            perm_p = g.get('permutation_p_value', -1)
            perm_str = f"{perm_p:.3f}" if perm_p >= 0 else "N/A"

            pass_str = "YES" if g.get('ALL_PASS') else "no"

            print(f"{sig_name:<25s} {hold_str:>4s} {m['n_trades']:>5d} {m['mean_return_pct']:>7.3f} "
                  f"{m['win_rate']:>5.1f} {m['sharpe']:>7.3f} {m['sortino']:>7.3f} {m['profit_factor']:>6.2f} "
                  f"{perm_str:>7s} {g['gates_passed']:>7s} {pass_str:>5s}")

            if g.get('ALL_PASS'):
                passing_combos.append((sig_name, hold_key, r))

    # Spread summary
    print("\n" + "=" * 80)
    print("CALL DEBIT SPREAD SIM (ATM / ATM+2%)")
    print("=" * 80)

    header2 = f"{'Signal':<25s} {'Hold':>4s} {'N':>5s} {'$PnL':>8s} {'Mean$/T':>8s} {'WR%':>5s} {'PF':>6s} {'MaxDD%':>7s} {'Final$':>8s}"
    print(header2)
    print("-" * len(header2))

    for sig_name in sorted(spread_results.keys()):
        for hold_key in sorted(spread_results[sig_name].keys()):
            s = spread_results[sig_name][hold_key]
            hold_str = hold_key.replace('hold_', '').replace('d', '')

            print(f"{sig_name:<25s} {hold_str:>4s} {s['n_trades']:>5d} {s['total_pnl']:>8.0f} "
                  f"{s['mean_pnl_per_trade']:>8.2f} {s['win_rate']:>5.1f} {s['profit_factor']:>6.2f} "
                  f"{s['max_drawdown_pct']:>7.1f} {s['final_equity']:>8.0f}")

    # Detail on passing combos
    if passing_combos:
        print("\n" + "=" * 80)
        print("DETAILED RESULTS FOR PASSING STRATEGIES")
        print("=" * 80)

        for sig_name, hold_key, r in passing_combos:
            print(f"\n--- {sig_name} / {hold_key} ---")

            print(f"  Metrics: {json.dumps(r['metrics'], indent=2)}")
            gates_safe = {k: bool(v) if isinstance(v, (np.bool_, np.generic)) else v for k, v in r['gates'].items()}
            print(f"  Gates: {json.dumps(gates_safe, indent=2)}")

            print(f"  Regimes:")
            for regime, data in r['regimes'].items():
                if data.get('mean_ret') is not None:
                    print(f"    {regime}: N={data['n']}, Mean={data['mean_ret']*100:.3f}%, WR={data['win_rate']*100:.1f}%")
                else:
                    print(f"    {regime}: N={data['n']} (insufficient)")

            print(f"  Sub-periods:")
            for period, data in r['sub_periods'].items():
                if data.get('mean_ret') is not None:
                    print(f"    {period}: N={data['n']}, Mean={data['mean_ret']*100:.3f}%, WR={data['win_rate']*100:.1f}%")
                else:
                    print(f"    {period}: N={data['n']} (insufficient)")

            if r['outlier']:
                print(f"  Outlier: orig_mean={r['outlier']['original_mean']*100:.3f}%, "
                      f"winsorized={r['outlier']['winsorized_mean']*100:.3f}%, "
                      f"change={r['outlier']['pct_change']:.1f}%")

            if r['drawdown']:
                print(f"  Drawdown: max={r['drawdown']['max_drawdown_pct']:.1f}%, "
                      f"max_consec_losses={r['drawdown']['max_consecutive_losses']}")
    else:
        print("\n  *** NO strategies passed ALL adversarial gates ***")

        # Show near-misses
        print("\n  Near-misses (passed 4+ gates):")
        for sig_name in sorted(all_results.keys()):
            for hold_key in sorted(all_results[sig_name].keys()):
                r = all_results[sig_name][hold_key]
                g = r['gates']
                passed = g.get('gates_passed', '0/0')
                p, t = passed.split('/')
                if int(p) >= 4:
                    hold_str = hold_key.replace('hold_', '').replace('d', '')
                    perm_p = g.get('permutation_p_value', -1)
                    print(f"    {sig_name} / {hold_str}d: {passed} gates, "
                          f"Mean={r['metrics']['mean_return_pct']:.3f}%, "
                          f"WR={r['metrics']['win_rate']:.1f}%, "
                          f"Sharpe={r['metrics']['sharpe']:.3f}, "
                          f"perm_p={perm_p:.3f}")
                    # Show which gates failed
                    for gate_name in ['permutation_p<0.05', 'win_rate>50%', 'profit_factor>1.0',
                                     'profitable_in_2+_subperiods', 'outlier_robust', 'exceeds_costs']:
                        if g.get(gate_name) == False:
                            print(f"      FAILED: {gate_name}")

    # ---- SAVE ----
    output_data = {
        'config': {
            'starting_capital': STARTING_CAPITAL,
            'risk_per_trade': RISK_PER_TRADE,
            'spread_width_pct': SPREAD_WIDTH_PCT,
            'hold_periods': HOLD_PERIODS,
            'n_permutations': N_PERMUTATIONS,
            'data_range': f"{df.index[0].date()} to {df.index[-1].date()}",
            'total_days': len(df),
        },
        'equity_results': {},
        'spread_results': {},
    }

    # Convert to serializable
    for sig_name, holds in all_results.items():
        output_data['equity_results'][sig_name] = {}
        for hold_key, r in holds.items():
            output_data['equity_results'][sig_name][hold_key] = r

    for sig_name, holds in spread_results.items():
        output_data['spread_results'][sig_name] = {}
        for hold_key, s in holds.items():
            output_data['spread_results'][sig_name][hold_key] = s

    def json_serializer(obj):
        if isinstance(obj, (np.bool_, np.generic)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    results_file = OUTPUT / "results.json"
    with open(results_file, 'w') as f:
        json.dump(output_data, f, indent=2, default=json_serializer)

    print(f"\n  Results saved to {results_file}")

    # ---- FINAL VERDICT ----
    print("\n" + "=" * 80)
    if passing_combos:
        print(f"VERDICT: {len(passing_combos)} signal/hold combos PASSED all adversarial gates.")
        print("Recommended for paper trading evaluation.")
    else:
        print("VERDICT: No combos passed ALL gates. See near-misses above.")
        print("Consider relaxing thresholds or combining signals.")
    print("=" * 80)


if __name__ == "__main__":
    main()
