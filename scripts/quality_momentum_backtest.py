#!/usr/bin/env python3
"""
Quality Momentum Backtest (Novy-Marx 2013 inspired)
====================================================
Combining quality (profitability/earnings) with momentum (price trend).
6 variants, 5-gate validation, permutation tests.

Universe: 24 large-cap stocks
OOT: Jan 2022 - Jul 2026
Capital: $645, $0 commission, 0.02% slippage per side
"""

import json, os, sys, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

# ─── CONFIG ─────────────────────────────────────────────────────────────
UNIVERSE = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD',
    'CRM', 'ADBE', 'NFLX', 'AVGO', 'COST', 'PEP', 'LLY', 'UNH',
    'V', 'MA', 'JPM', 'HD', 'INTC', 'MU', 'QCOM', 'PYPL'
]
START_DATE = '2020-07-01'   # need lookback for 12m momentum
END_DATE   = '2026-07-30'
OOT_START  = '2022-01-01'
ACCOUNT    = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% per side
RF_ANNUAL  = 0.045
N_PERM     = 1000
RESULTS_PATH = '/home/jupiter/Lvl3Quant/data/quality_momentum_results.json'

# ─── DATA DOWNLOAD ──────────────────────────────────────────────────────
print("Downloading price data (batch)...")
tickers_needed = UNIVERSE + ['SPY']
raw = yf.download(tickers_needed, start=START_DATE, end=END_DATE, progress=True, auto_adjust=True, threads=True)
# yfinance returns MultiIndex columns: (Price, Ticker)
if isinstance(raw.columns, pd.MultiIndex):
    price_df = raw['Close'].copy()
else:
    price_df = raw[['Close']].copy()
    price_df.columns = [tickers_needed[0]]

# Drop tickers with insufficient data
good = [c for c in price_df.columns if price_df[c].notna().sum() > 100]
price_df = price_df[good].ffill().dropna(how='all')
missing = set(tickers_needed) - set(good)
if missing:
    print(f"  WARNING: Insufficient data for {missing}")
print(f"  Price data: {price_df.index[0].date()} to {price_df.index[-1].date()}, {len(price_df)} days, {len(price_df.columns)} tickers")

# SPY for regime filter
spy_close = price_df['SPY'].copy()
spy_sma200 = spy_close.rolling(200).mean()
bull_regime = spy_close > spy_sma200  # True = bull

# ─── EARNINGS PROXY ─────────────────────────────────────────────────────
# Try yfinance earnings; fallback to weekly return > 3% as earnings beat proxy
print("Building earnings beat proxy...")

def get_earnings_dates_yf(ticker):
    """Try to get actual earnings dates from yfinance."""
    try:
        tk = yf.Ticker(ticker)
        ed = tk.earnings_dates
        if ed is not None and len(ed) > 0:
            # ed index is earnings date, has 'Surprise(%)' column sometimes
            return ed
    except Exception:
        pass
    return None

# Build earnings beat signals: for each stock, at each month-end,
# did it have a positive earnings surprise in the last 90 days?
earnings_beat = {}  # ticker -> Series of dates where beat occurred
for idx, t in enumerate(UNIVERSE):
    if t not in price_df.columns:
        continue
    print(f"  [{idx+1}/{len(UNIVERSE)}] Fetching earnings for {t}...", end=' ', flush=True)
    ed = get_earnings_dates_yf(t)
    if ed is not None and 'Surprise(%)' in ed.columns:
        # Use actual surprise data
        beats = ed[ed['Surprise(%)'] > 0].index
        earnings_beat[t] = pd.DatetimeIndex([d.tz_localize(None) if d.tzinfo else d for d in beats])
        print(f"{len(beats)} beats from yfinance")
    else:
        # Proxy: weekly return > 3% suggests earnings beat (gap up)
        if t in price_df.columns:
            weekly_ret = price_df[t].pct_change(5)
            big_up = weekly_ret[weekly_ret > 0.03].index
            earnings_beat[t] = big_up
            print(f"{len(big_up)} proxy beats")

def had_earnings_beat(ticker, date, lookback_days=90):
    """Check if ticker had an earnings beat within lookback_days before date."""
    if ticker not in earnings_beat or len(earnings_beat[ticker]) == 0:
        return False
    beat_dates = earnings_beat[ticker]
    cutoff = date - pd.Timedelta(days=lookback_days)
    mask = (beat_dates >= cutoff) & (beat_dates <= date)
    return mask.any()

def earnings_beat_magnitude(ticker, date, lookback_days=90):
    """Return the magnitude of the best earnings beat in lookback period."""
    if ticker not in price_df.columns:
        return 0.0
    cutoff = date - pd.Timedelta(days=lookback_days)
    sub = price_df[ticker].loc[cutoff:date]
    if len(sub) < 5:
        return 0.0
    weekly_ret = sub.pct_change(5).dropna()
    if len(weekly_ret) == 0:
        return 0.0
    best = weekly_ret.max()
    return max(best, 0.0)

# ─── HELPER FUNCTIONS ───────────────────────────────────────────────────
def get_monthly_rebal_dates(start, end):
    """Get month-end business days for rebalancing."""
    dates = price_df.loc[start:end].index
    monthly = dates.to_period('M')
    rebal = []
    for period in monthly.unique():
        mask = dates.to_period('M') == period
        month_dates = dates[mask]
        if len(month_dates) > 0:
            rebal.append(month_dates[-1])
    return rebal

def compute_returns(price_series, n_days):
    """Compute n-day return."""
    return price_series.pct_change(n_days)

def zscore_series(s):
    """Z-score a series, handling NaN."""
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return s * 0
    return (s - s.mean()) / s.std()

def simulate_strategy(selection_func, name, n_positions=5):
    """
    Generic monthly rebalance simulator.
    selection_func(date) -> list of tickers to hold (or empty for cash).
    Returns dict with equity curve and metrics.
    """
    rebal_dates = get_monthly_rebal_dates(OOT_START, END_DATE)

    equity = ACCOUNT
    equity_curve = []
    holdings = []
    trades = 0
    daily_returns = []

    current_positions = {}  # ticker -> shares

    for i, rdate in enumerate(rebal_dates):
        # Get new selections
        selected = selection_func(rdate)
        if not selected:
            selected = []

        # Calculate returns from previous rebal to this one
        if i > 0:
            prev_date = rebal_dates[i-1]
            # Get prices at prev and current dates
            for t, shares in current_positions.items():
                if t in price_df.columns:
                    p_prev = price_df[t].asof(prev_date)
                    p_now = price_df[t].asof(rdate)
                    if pd.notna(p_prev) and pd.notna(p_now) and p_prev > 0:
                        ret = (p_now / p_prev) - 1
                        weight = shares * p_prev / equity if equity > 0 else 0
                        equity += shares * (p_now - p_prev)

            # Compute daily returns for this period
            period_prices = price_df.loc[prev_date:rdate]
            if len(period_prices) > 1 and current_positions:
                port_value_series = pd.Series(0.0, index=period_prices.index)
                for t, shares in current_positions.items():
                    if t in period_prices.columns:
                        port_value_series += shares * period_prices[t]
                # Add cash component
                invested = sum(shares * price_df[t].asof(prev_date)
                             for t, shares in current_positions.items()
                             if t in price_df.columns and pd.notna(price_df[t].asof(prev_date)))
                cash = max(equity - invested, 0) if i == 1 else 0
                port_daily_ret = port_value_series.pct_change().dropna()
                daily_returns.extend(port_daily_ret.values)

        # Rebalance: equal weight into selected stocks
        # Apply slippage on trades
        new_positions = {}
        if selected:
            n_pos = len(selected)
            alloc_per = equity / n_pos if equity > 0 else 0
            for t in selected:
                if t in price_df.columns:
                    p = price_df[t].asof(rdate)
                    if pd.notna(p) and p > 0:
                        # Slippage cost
                        slippage = alloc_per * SLIPPAGE_PCT
                        effective_alloc = alloc_per - slippage
                        shares = effective_alloc / p
                        new_positions[t] = shares
                        if t not in current_positions:
                            trades += 1
                        elif abs(current_positions.get(t, 0) - shares) / max(shares, 1e-9) > 0.1:
                            trades += 1

        current_positions = new_positions
        equity_curve.append({'date': str(rdate.date()), 'equity': round(equity, 2)})

    # Final mark-to-market
    if rebal_dates:
        last_rdate = rebal_dates[-1]
        last_price_date = price_df.index[-1]
        if last_price_date > last_rdate:
            for t, shares in current_positions.items():
                if t in price_df.columns:
                    p_prev = price_df[t].asof(last_rdate)
                    p_now = price_df[t].asof(last_price_date)
                    if pd.notna(p_prev) and pd.notna(p_now) and p_prev > 0:
                        equity += shares * (p_now - p_prev)
            equity_curve.append({'date': str(last_price_date.date()), 'equity': round(equity, 2)})

    # Compute metrics from daily returns
    daily_returns = [r for r in daily_returns if np.isfinite(r)]

    if len(daily_returns) < 10:
        return {
            'name': name, 'final_equity': round(equity, 2),
            'total_return_pct': round((equity / ACCOUNT - 1) * 100, 2),
            'sharpe': 0.0, 'sortino': 0.0, 'max_dd_pct': 0.0,
            'profit_factor': 0.0, 'win_rate': 0.0, 'trades': trades,
            'equity_curve': equity_curve, 'daily_returns': daily_returns,
            'gates': {}
        }

    dr = np.array(daily_returns)
    annual_factor = np.sqrt(252)
    rf_daily = RF_ANNUAL / 252
    excess = dr - rf_daily
    sharpe = (np.mean(excess) / np.std(excess) * annual_factor) if np.std(excess) > 0 else 0

    downside = excess[excess < 0]
    sortino = (np.mean(excess) / np.std(downside) * annual_factor) if len(downside) > 0 and np.std(downside) > 0 else 0

    # Max drawdown from equity curve
    eq_values = [e['equity'] for e in equity_curve]
    peak = eq_values[0]
    max_dd = 0
    for v in eq_values:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Profit factor from monthly returns
    monthly_rets = []
    for i in range(1, len(equity_curve)):
        r = equity_curve[i]['equity'] / equity_curve[i-1]['equity'] - 1
        monthly_rets.append(r)

    gains = sum(r for r in monthly_rets if r > 0)
    losses = abs(sum(r for r in monthly_rets if r < 0))
    profit_factor = gains / losses if losses > 0 else (999 if gains > 0 else 0)

    win_rate = sum(1 for r in monthly_rets if r > 0) / len(monthly_rets) * 100 if monthly_rets else 0

    return {
        'name': name,
        'final_equity': round(equity, 2),
        'total_return_pct': round((equity / ACCOUNT - 1) * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_dd_pct': round(max_dd * 100, 2),
        'profit_factor': round(profit_factor, 2),
        'win_rate': round(win_rate, 1),
        'trades': trades,
        'equity_curve': equity_curve,
        'daily_returns': daily_returns,
        'monthly_returns': monthly_rets,
        'gates': {}
    }

def regime_sharpe(result):
    """Compute Sharpe in bull vs bear regimes."""
    eq = result['equity_curve']
    if len(eq) < 4:
        return 0.0, 0.0

    bull_rets, bear_rets = [], []
    for i in range(1, len(eq)):
        d = pd.Timestamp(eq[i]['date'])
        r = eq[i]['equity'] / eq[i-1]['equity'] - 1
        if d in bull_regime.index and bull_regime.asof(d):
            bull_rets.append(r)
        else:
            bear_rets.append(r)

    rf_monthly = RF_ANNUAL / 12
    def monthly_sharpe(rets):
        if len(rets) < 3:
            return 0.0
        excess = np.array(rets) - rf_monthly
        if np.std(excess) == 0:
            return 0.0
        return np.mean(excess) / np.std(excess) * np.sqrt(12)

    return monthly_sharpe(bull_rets), monthly_sharpe(bear_rets)

def permutation_test(selection_func, actual_sharpe, n_perm=N_PERM, n_positions=5):
    """
    Permutation test: shuffle which stocks get selected each month.
    Returns p-value (fraction of perms with Sharpe >= actual).
    """
    rebal_dates = get_monthly_rebal_dates(OOT_START, END_DATE)
    avail_tickers = [t for t in UNIVERSE if t in price_df.columns]

    perm_sharpes = []
    for _ in range(n_perm):
        # Random selection each month
        def random_select(date):
            n = min(n_positions, len(avail_tickers))
            return list(np.random.choice(avail_tickers, n, replace=False))

        res = simulate_strategy(random_select, "perm", n_positions=n_positions)
        perm_sharpes.append(res['sharpe'])

    p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
    return p_value

# ─── STRATEGY DEFINITIONS ───────────────────────────────────────────────

# A. 12-1 Momentum: top 5 by 12-month return, skip last month
def strategy_a_select(date):
    scores = {}
    for t in UNIVERSE:
        if t not in price_df.columns or t == 'SPY':
            continue
        d12m = date - pd.Timedelta(days=365)
        d1m = date - pd.Timedelta(days=30)
        p12m = price_df[t].asof(d12m)
        p1m = price_df[t].asof(d1m)
        if pd.notna(p12m) and pd.notna(p1m) and p12m > 0:
            ret = (p1m / p12m) - 1  # 12-1 month return
            scores[t] = ret
    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return ranked[:5]

# B. Dual Momentum: positive absolute + top 5 relative
def strategy_b_select(date):
    scores = {}
    for t in UNIVERSE:
        if t not in price_df.columns or t == 'SPY':
            continue
        d12m = date - pd.Timedelta(days=365)
        p12m = price_df[t].asof(d12m)
        p_now = price_df[t].asof(date)
        if pd.notna(p12m) and pd.notna(p_now) and p12m > 0:
            ret = (p_now / p12m) - 1
            if ret > 0:  # positive absolute momentum
                scores[t] = ret
    if len(scores) < 3:
        return []  # cash when < 3 qualify
    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return ranked[:5]

# C. Momentum + Earnings Quality: top 5 by 6m return WITH positive earnings surprise
def strategy_c_select(date):
    scores = {}
    for t in UNIVERSE:
        if t not in price_df.columns or t == 'SPY':
            continue
        d6m = date - pd.Timedelta(days=182)
        p6m = price_df[t].asof(d6m)
        p_now = price_df[t].asof(date)
        if pd.notna(p6m) and pd.notna(p_now) and p6m > 0:
            ret = (p_now / p6m) - 1
            if had_earnings_beat(t, date, lookback_days=90):
                scores[t] = ret
    ranked = sorted(scores, key=lambda x: scores[x], reverse=True)
    return ranked[:5]

# D. Trend-Filtered Momentum: like A but only in bull market
def strategy_d_select(date):
    if date in bull_regime.index:
        is_bull = bull_regime.asof(date)
    else:
        is_bull = bull_regime.asof(date)
    if not is_bull:
        return []  # cash in bear market
    return strategy_a_select(date)

# E. Low Vol Momentum: positive 6m momentum, pick 5 with lowest 30d vol
def strategy_e_select(date):
    candidates = {}
    for t in UNIVERSE:
        if t not in price_df.columns or t == 'SPY':
            continue
        d6m = date - pd.Timedelta(days=182)
        p6m = price_df[t].asof(d6m)
        p_now = price_df[t].asof(date)
        if pd.notna(p6m) and pd.notna(p_now) and p6m > 0:
            ret = (p_now / p6m) - 1
            if ret > 0:  # positive momentum filter
                # 30-day realized vol
                d30 = date - pd.Timedelta(days=30)
                sub = price_df[t].loc[d30:date]
                if len(sub) > 5:
                    vol = sub.pct_change().std() * np.sqrt(252)
                    candidates[t] = vol
    # Sort by LOWEST vol
    ranked = sorted(candidates, key=lambda x: candidates[x])
    return ranked[:5]

# F. Concentrated Best: top 3 by combined z-score
def strategy_f_select(date):
    mom_scores = {}
    eb_scores = {}
    vol_scores = {}

    for t in UNIVERSE:
        if t not in price_df.columns or t == 'SPY':
            continue
        # 6m return
        d6m = date - pd.Timedelta(days=182)
        p6m = price_df[t].asof(d6m)
        p_now = price_df[t].asof(date)
        if pd.notna(p6m) and pd.notna(p_now) and p6m > 0:
            mom_scores[t] = (p_now / p6m) - 1

        # Earnings beat magnitude
        eb_scores[t] = earnings_beat_magnitude(t, date, lookback_days=90)

        # 30d vol (negative = lower is better)
        d30 = date - pd.Timedelta(days=30)
        sub = price_df[t].loc[d30:date]
        if len(sub) > 5:
            vol_scores[t] = sub.pct_change().std() * np.sqrt(252)

    # Only tickers with all three scores
    common = set(mom_scores) & set(eb_scores) & set(vol_scores)
    if len(common) < 3:
        return list(common)

    # Z-score each
    mom_s = pd.Series({t: mom_scores[t] for t in common})
    eb_s = pd.Series({t: eb_scores[t] for t in common})
    vol_s = pd.Series({t: -vol_scores[t] for t in common})  # negative vol = better

    z_mom = zscore_series(mom_s)
    z_eb = zscore_series(eb_s)
    z_vol = zscore_series(vol_s)

    # Combined: 0.5 * z(6m_return) + 0.3 * z(earnings_beat) + 0.2 * z(-vol)
    combined = 0.5 * z_mom + 0.3 * z_eb + 0.2 * z_vol
    combined = combined.dropna().sort_values(ascending=False)
    return list(combined.index[:3])

# ─── RUN ALL STRATEGIES ─────────────────────────────────────────────────
strategies = [
    ('A_12_1_Momentum', strategy_a_select, 5),
    ('B_Dual_Momentum', strategy_b_select, 5),
    ('C_Momentum_Earnings_Quality', strategy_c_select, 5),
    ('D_Trend_Filtered_Momentum', strategy_d_select, 5),
    ('E_Low_Vol_Momentum', strategy_e_select, 5),
    ('F_Concentrated_Best', strategy_f_select, 3),
]

all_results = {}
print("\n" + "="*80)
print("QUALITY MOMENTUM BACKTEST — 6 VARIANTS")
print(f"Universe: {len(UNIVERSE)} stocks | OOT: {OOT_START} to {END_DATE}")
print(f"Capital: ${ACCOUNT} | Slippage: {SLIPPAGE_PCT*100:.2f}% per side")
print("="*80)

for sname, sfunc, npos in strategies:
    print(f"\n--- Running {sname} ---")
    result = simulate_strategy(sfunc, sname, n_positions=npos)

    # Regime analysis
    bull_sharpe, bear_sharpe = regime_sharpe(result)
    result['bull_sharpe'] = round(bull_sharpe, 3)
    result['bear_sharpe'] = round(bear_sharpe, 3)

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0
    result['regime_gap'] = round(regime_gap, 3)

    # Permutation test
    print(f"  Running permutation test ({N_PERM} iterations)...")
    p_value = permutation_test(sfunc, result['sharpe'], n_perm=N_PERM, n_positions=npos)
    result['perm_p_value'] = round(p_value, 4)

    # 5-Gate Validation
    gates = {
        'sharpe_gt_0.5': result['sharpe'] > 0.5,
        'perm_p_lt_0.05': p_value < 0.05,
        'regime_gap_lt_0.5': regime_gap < 0.5,
        'max_dd_gt_neg50': result['max_dd_pct'] > -50,
        'trades_gte_20': result['trades'] >= 20,
    }
    result['gates'] = gates
    result['gates_passed'] = sum(gates.values())
    result['all_gates_passed'] = all(gates.values())

    # Clean up for JSON serialization
    result['daily_returns'] = []  # too large for JSON
    result.pop('monthly_returns', None)

    all_results[sname] = result

    # Print summary
    print(f"  Final Equity: ${result['final_equity']:.2f} | Return: {result['total_return_pct']:.1f}%")
    print(f"  Sharpe: {result['sharpe']:.3f} | Sortino: {result['sortino']:.3f} | MaxDD: {result['max_dd_pct']:.1f}%")
    print(f"  PF: {result['profit_factor']:.2f} | WR: {result['win_rate']:.1f}% | Trades: {result['trades']}")
    print(f"  Bull Sharpe: {result['bull_sharpe']:.3f} | Bear Sharpe: {result['bear_sharpe']:.3f} | Regime Gap: {result['regime_gap']:.3f}")
    print(f"  Perm p-value: {result['perm_p_value']:.4f}")
    print(f"  Gates: {result['gates_passed']}/5 {'✓ ALL PASSED' if result['all_gates_passed'] else '✗ FAILED'}")
    for gname, gval in gates.items():
        status = '✓' if gval else '✗'
        print(f"    {status} {gname}")

# ─── SUMMARY ────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
header = f"{'Strategy':<32} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'PF':>5} {'WR%':>5} {'Trades':>6} {'Gates':>6}"
print(header)
print("-" * len(header))
for sname in all_results:
    r = all_results[sname]
    gates_str = f"{r['gates_passed']}/5"
    if r['all_gates_passed']:
        gates_str += " ✓"
    print(f"{sname:<32} {r['total_return_pct']:>8.1f} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['max_dd_pct']:>7.1f} {r['profit_factor']:>5.2f} {r['win_rate']:>5.1f} {r['trades']:>6} {gates_str:>6}")

# Identify best
best = max(all_results.values(), key=lambda x: x['sharpe'])
print(f"\nBest by Sharpe: {best['name']} (Sharpe={best['sharpe']:.3f})")

passed = [r for r in all_results.values() if r['all_gates_passed']]
print(f"Strategies passing all 5 gates: {len(passed)}")
for r in passed:
    print(f"  - {r['name']}: Sharpe={r['sharpe']:.3f}, Return={r['total_return_pct']:.1f}%")

# ─── SAVE RESULTS ────────────────────────────────────────────────────────
output = {
    'metadata': {
        'strategy': 'Quality Momentum (Novy-Marx 2013)',
        'run_date': str(datetime.now()),
        'universe_size': len(UNIVERSE),
        'oot_period': f'{OOT_START} to {END_DATE}',
        'starting_capital': ACCOUNT,
        'slippage_pct': SLIPPAGE_PCT,
        'n_permutations': N_PERM,
    },
    'variants': {}
}

for sname, r in all_results.items():
    output['variants'][sname] = {
        'final_equity': r['final_equity'],
        'total_return_pct': r['total_return_pct'],
        'sharpe': r['sharpe'],
        'sortino': r['sortino'],
        'max_dd_pct': r['max_dd_pct'],
        'profit_factor': r['profit_factor'],
        'win_rate': r['win_rate'],
        'trades': r['trades'],
        'bull_sharpe': r['bull_sharpe'],
        'bear_sharpe': r['bear_sharpe'],
        'regime_gap': r['regime_gap'],
        'perm_p_value': r['perm_p_value'],
        'gates': r['gates'],
        'gates_passed': r['gates_passed'],
        'all_gates_passed': r['all_gates_passed'],
        'equity_curve': r['equity_curve'],
    }

with open(RESULTS_PATH, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to {RESULTS_PATH}")
print("Done.")
