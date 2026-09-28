#!/usr/bin/env python3
"""
Volatility Risk Premium (VRP) Harvesting Backtest
==================================================
Systematically profits from the well-documented tendency of implied
volatility (VIX) to exceed realized volatility (SPY 20-day).

VRP = VIX - RealizedVol(SPY, 20d, annualized)

Variants:
  A) VRP Timing on QQQ (VRP>5 → QQQ, VRP<0 → SHY, hysteresis)
  B) VRP + Momentum Blend (VRP>5 → top-3 momentum growth, VRP<2 → GLD)
  C) VRP Percentile Thresholds (rolling 252d percentile)
  D) VRP Mean Reversion (spike>10 → buy 5d later, drop<-3 → GLD)
  E) VRP-Scaled Position Sizing (always in QQQ, size by VRP)
  F) VRP + RSI Combo (VRP>5 regime + RSI(5)<30 entry)

5-Gate: Sharpe>0.5, Perm p<0.05, Regime gap<0.5, MaxDD>-50%, >=20 trades.
OOT: Jan 2022 - Jul 2026. $645 account. Robinhood ($0 commission, 0.02% slippage).
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
START = "2020-01-01"    # extra lookback for 252-day rolling + 200-SMA
END = "2026-07-31"
OOT_START = "2022-01-01"
N_PERM = 1000

GROWTH_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD"]
ALL_TICKERS = sorted(set(GROWTH_TICKERS + [
    "SPY", "QQQ", "^VIX", "SVXY", "UVXY", "GLD", "TLT", "SHY"
]))

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))
qqq_close = closes.get("QQQ", pd.Series(dtype=float))
vix_close = closes.get("^VIX", pd.Series(dtype=float))
gld_close = closes.get("GLD", pd.Series(dtype=float))
shy_close = closes.get("SHY", pd.Series(dtype=float))

loaded = sum(1 for t in ALL_TICKERS if len(closes.get(t, [])) > 252)
print(f"  Tickers with sufficient data: {loaded}/{len(ALL_TICKERS)}")
print(f"  SPY rows: {len(spy_close)}, VIX rows: {len(vix_close)}, QQQ rows: {len(qqq_close)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def sma(series, period):
    return series.rolling(period).mean()


def realized_vol_annualized(series, window=20):
    """Annualized realized volatility from daily log returns."""
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252) * 100  # as percentage points


# ── Pre-compute VRP and indicators ───────────────────────────────────────
print("Computing VRP and indicators ...")

# VRP = VIX - Realized Vol (SPY, 20d annualized)
rv_spy = realized_vol_annualized(spy_close, 20)

# Align VIX and RV on common dates
common_idx = vix_close.index.intersection(rv_spy.index)
vix_aligned = vix_close.loc[common_idx]
rv_aligned = rv_spy.loc[common_idx]
vrp = vix_aligned - rv_aligned
vrp = vrp.dropna()

# Rolling VRP percentile (252-day)
vrp_pctile = vrp.rolling(252, min_periods=63).rank(pct=True) * 100

# SPY 200-SMA for regime classification
spy_sma200 = sma(spy_close, 200)
spy_regime = (spy_close > spy_sma200).reindex(vrp.index).fillna(True)  # True=Bull

# RSI(5) on QQQ
qqq_rsi5 = calc_rsi(qqq_close, 5)

# Momentum: 63-day return for growth stocks
momentum_63 = {}
for t in GROWTH_TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) > 63:
        momentum_63[t] = c.pct_change(63)

# Daily returns for assets
returns = {}
for t in ALL_TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) > 1:
        r = c.pct_change()
        returns[t] = r

print(f"  VRP computed: {len(vrp)} days, range [{vrp.min():.1f}, {vrp.max():.1f}]")
print(f"  VRP mean: {vrp.mean():.2f}, median: {vrp.median():.2f}")


# ── Backtest Engine ──────────────────────────────────────────────────────
def run_backtest(variant_name, signal_func, oot_start=OOT_START, vrp_series=None):
    """
    Generic backtest engine.
    signal_func(date, state) -> dict with:
        'allocations': dict of ticker -> weight (must sum to <= 1.0)
    state is a mutable dict for the strategy to track its own state.
    """
    if vrp_series is None:
        vrp_series = vrp

    oot_dates = vrp_series.index[vrp_series.index >= oot_start]
    if len(oot_dates) == 0:
        return None

    equity = CAPITAL
    equity_curve = []
    trades = []
    state = {
        'current_alloc': {},
        'equity': CAPITAL,
        'last_rebalance': None,
    }

    prev_alloc = {}

    for i, date in enumerate(oot_dates):
        # Get signal
        result = signal_func(date, state)
        new_alloc = result.get('allocations', {})

        # Calculate daily return based on current allocation
        daily_ret = 0.0
        for ticker, weight in prev_alloc.items():
            if ticker in returns and date in returns[ticker].index:
                r = returns[ticker].loc[date]
                if np.isfinite(r):
                    daily_ret += weight * r

        # Apply slippage on rebalance
        if new_alloc != prev_alloc:
            turnover = 0.0
            all_tickers_involved = set(list(new_alloc.keys()) + list(prev_alloc.keys()))
            for t in all_tickers_involved:
                old_w = prev_alloc.get(t, 0)
                new_w = new_alloc.get(t, 0)
                turnover += abs(new_w - old_w)
            slippage_cost = turnover * SLIPPAGE_PCT
            daily_ret -= slippage_cost

            # Record trade
            if prev_alloc != new_alloc:
                trades.append({
                    'date': str(date.date()) if hasattr(date, 'date') else str(date),
                    'from': dict(prev_alloc),
                    'to': dict(new_alloc),
                    'equity': equity,
                })

        equity *= (1 + daily_ret)
        equity_curve.append({'date': date, 'equity': equity})
        state['equity'] = equity
        prev_alloc = dict(new_alloc)

    if len(equity_curve) == 0:
        return None

    eq_df = pd.DataFrame(equity_curve).set_index('date')
    eq_df['returns'] = eq_df['equity'].pct_change()

    return {
        'equity_curve': eq_df,
        'trades': trades,
        'final_equity': equity,
    }


def compute_metrics(result):
    """Compute performance metrics from backtest result."""
    if result is None:
        return None

    eq_df = result['equity_curve']
    rets = eq_df['returns'].dropna()

    if len(rets) < 20:
        return None

    total_ret = (result['final_equity'] / CAPITAL) - 1
    ann_ret = (1 + total_ret) ** (252 / len(rets)) - 1 if len(rets) > 0 else 0
    ann_vol = rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else 0
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()

    # Win rate (of trade periods — days with positive return when allocated)
    win_days = (rets > 0).sum()
    total_days = (rets != 0).sum()
    win_rate = win_days / total_days if total_days > 0 else 0

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    n_trades = len(result['trades'])

    return {
        'total_return_pct': round(total_ret * 100, 2),
        'ann_return_pct': round(ann_ret * 100, 2),
        'ann_volatility_pct': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown_pct': round(max_dd * 100, 2),
        'profit_factor': round(profit_factor, 3),
        'win_rate_pct': round(win_rate * 100, 1),
        'n_trades': n_trades,
        'final_equity': round(result['final_equity'], 2),
    }


def compute_regime_metrics(result, regime_series):
    """Compute metrics split by bull/bear regime."""
    if result is None:
        return None

    eq_df = result['equity_curve']
    rets = eq_df['returns'].dropna()

    # Align regime with returns
    regime_aligned = regime_series.reindex(rets.index).fillna(True)

    bull_rets = rets[regime_aligned == True]
    bear_rets = rets[regime_aligned == False]

    def _sharpe(r):
        if len(r) < 10 or r.std() == 0:
            return 0.0
        return (r.mean() * 252) / (r.std() * np.sqrt(252))

    bull_sharpe = _sharpe(bull_rets)
    bear_sharpe = _sharpe(bear_rets)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 0.001)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'bull_days': int(len(bull_rets)),
        'bear_days': int(len(bear_rets)),
    }


def permutation_test(variant_func, n_perm=N_PERM):
    """
    Permutation test: offset VRP signal by random 1-60 day shift.
    Returns p-value (fraction of permutations with Sharpe >= real).
    """
    # Run real backtest
    real_result = run_backtest("real", variant_func)
    if real_result is None:
        return 1.0, 0

    real_metrics = compute_metrics(real_result)
    if real_metrics is None:
        return 1.0, 0
    real_sharpe = real_metrics['sharpe']

    better_count = 0
    valid_perms = 0

    for i in range(n_perm):
        # Create shifted VRP
        shift = np.random.randint(1, 61)
        vrp_shifted = vrp.shift(shift).dropna()

        # Create a closure with shifted VRP
        def make_shifted_func(shift_val):
            def shifted_func(date, state):
                # Replace vrp lookups with shifted version
                if date in vrp_shifted.index:
                    state['_vrp_override'] = vrp_shifted.loc[date]
                return variant_func(date, state)
            return shifted_func

        shifted_result = run_backtest("perm", make_shifted_func(shift))
        if shifted_result is None:
            continue

        perm_metrics = compute_metrics(shifted_result)
        if perm_metrics is None:
            continue

        valid_perms += 1
        if perm_metrics['sharpe'] >= real_sharpe:
            better_count += 1

    p_value = (better_count + 1) / (valid_perms + 1) if valid_perms > 0 else 1.0
    return p_value, valid_perms


def check_gates(metrics, regime, perm_p, n_trades):
    """Check 5-gate validation."""
    gates = {}
    gates['sharpe_gt_0.5'] = metrics['sharpe'] > 0.5
    gates['perm_p_lt_0.05'] = perm_p < 0.05
    gates['regime_gap_lt_0.5'] = regime['regime_gap'] < 0.5
    gates['max_dd_gt_neg50'] = metrics['max_drawdown_pct'] > -50.0
    gates['n_trades_gte_20'] = n_trades >= 20
    gates['all_pass'] = all(gates.values())
    return gates


# ── VARIANT A: VRP Timing on QQQ ─────────────────────────────────────────
def variant_a_signal(date, state):
    """VRP > 5 → QQQ. VRP < 0 → SHY. 0-5 → hold (hysteresis). Monthly review."""
    v = state.get('_vrp_override', vrp.get(date, None))
    if v is None or not np.isfinite(v):
        return {'allocations': state.get('current_alloc', {'SHY': 1.0})}

    # Monthly rebalance check
    last_reb = state.get('last_rebalance', None)
    if last_reb is not None:
        if hasattr(date, 'month') and hasattr(last_reb, 'month'):
            if date.year == last_reb.year and date.month == last_reb.month:
                return {'allocations': state.get('current_alloc', {'SHY': 1.0})}

    current = state.get('current_alloc', {'SHY': 1.0})

    if v > 5:
        alloc = {'QQQ': 1.0}
    elif v < 0:
        alloc = {'SHY': 1.0}
    else:
        alloc = current  # hysteresis

    state['current_alloc'] = alloc
    state['last_rebalance'] = date
    return {'allocations': alloc}


# ── VARIANT B: VRP + Momentum Blend ──────────────────────────────────────
def variant_b_signal(date, state):
    """VRP > 5 → top-3 momentum growth stocks. VRP < 2 → GLD."""
    v = state.get('_vrp_override', vrp.get(date, None))
    if v is None or not np.isfinite(v):
        return {'allocations': state.get('current_alloc', {'GLD': 1.0})}

    # Monthly rebalance
    last_reb = state.get('last_rebalance', None)
    if last_reb is not None:
        if hasattr(date, 'month') and hasattr(last_reb, 'month'):
            if date.year == last_reb.year and date.month == last_reb.month:
                return {'allocations': state.get('current_alloc', {'GLD': 1.0})}

    if v > 5:
        # Pick top-3 momentum growth stocks
        mom_scores = {}
        for t in GROWTH_TICKERS:
            if t in momentum_63 and date in momentum_63[t].index:
                m = momentum_63[t].loc[date]
                if np.isfinite(m):
                    mom_scores[t] = m

        if len(mom_scores) >= 3:
            top3 = sorted(mom_scores, key=mom_scores.get, reverse=True)[:3]
            alloc = {t: 1.0/3 for t in top3}
        else:
            alloc = {'QQQ': 1.0}  # fallback
    elif v < 2:
        alloc = {'GLD': 1.0}
    else:
        alloc = state.get('current_alloc', {'GLD': 1.0})

    state['current_alloc'] = alloc
    state['last_rebalance'] = date
    return {'allocations': alloc}


# ── VARIANT C: VRP Percentile Thresholds ─────────────────────────────────
def variant_c_signal(date, state):
    """VRP in top 20% (extreme fear premium) → QQQ. Bottom 20% → GLD. Middle → hold."""
    v_pct = vrp_pctile.get(date, None)
    if v_pct is None or not np.isfinite(v_pct):
        return {'allocations': state.get('current_alloc', {'GLD': 1.0})}

    # Monthly rebalance
    last_reb = state.get('last_rebalance', None)
    if last_reb is not None:
        if hasattr(date, 'month') and hasattr(last_reb, 'month'):
            if date.year == last_reb.year and date.month == last_reb.month:
                return {'allocations': state.get('current_alloc', {'GLD': 1.0})}

    if v_pct >= 80:
        alloc = {'QQQ': 1.0}
    elif v_pct <= 20:
        alloc = {'GLD': 1.0}
    else:
        alloc = state.get('current_alloc', {'GLD': 1.0})

    state['current_alloc'] = alloc
    state['last_rebalance'] = date
    return {'allocations': alloc}


# ── VARIANT D: VRP Mean Reversion ────────────────────────────────────────
def variant_d_signal(date, state):
    """VRP spikes > 10 → buy QQQ 5 days later. VRP < -3 → GLD. Wait otherwise."""
    v = state.get('_vrp_override', vrp.get(date, None))
    if v is None or not np.isfinite(v):
        return {'allocations': state.get('current_alloc', {'GLD': 1.0})}

    # Track spike dates
    if 'spike_dates' not in state:
        state['spike_dates'] = []
    if 'in_qqq_since' not in state:
        state['in_qqq_since'] = None

    current = state.get('current_alloc', {'GLD': 1.0})

    if v > 10:
        state['spike_dates'].append(date)

    if v < -3:
        alloc = {'GLD': 1.0}
        state['in_qqq_since'] = None
    else:
        # Check if we should enter QQQ (5 days after a spike)
        alloc = current
        for spike_date in list(state['spike_dates']):
            # Count business days since spike
            dates_between = vrp.index[(vrp.index > spike_date) & (vrp.index <= date)]
            if len(dates_between) >= 5:
                alloc = {'QQQ': 1.0}
                state['spike_dates'].remove(spike_date)
                state['in_qqq_since'] = date
                break

    state['current_alloc'] = alloc
    return {'allocations': alloc}


# ── VARIANT E: VRP-Scaled Position Sizing ────────────────────────────────
def variant_e_signal(date, state):
    """Always in QQQ, scale by VRP. Weekly rebalance."""
    v = state.get('_vrp_override', vrp.get(date, None))
    if v is None or not np.isfinite(v):
        return {'allocations': state.get('current_alloc', {'QQQ': 0.5, 'SHY': 0.5})}

    # Weekly rebalance (every 5 trading days)
    if 'day_count' not in state:
        state['day_count'] = 0
    state['day_count'] += 1

    if state['day_count'] % 5 != 1 and state.get('current_alloc'):
        return {'allocations': state['current_alloc']}

    if v > 8:
        alloc = {'QQQ': 1.0}
    elif v > 4:
        alloc = {'QQQ': 0.75, 'SHY': 0.25}
    elif v > 0:
        alloc = {'QQQ': 0.50, 'SHY': 0.50}
    else:
        alloc = {'QQQ': 0.25, 'SHY': 0.75}

    state['current_alloc'] = alloc
    return {'allocations': alloc}


# ── VARIANT F: VRP + RSI Combo ───────────────────────────────────────────
def variant_f_signal(date, state):
    """VRP > 5 = regime filter. Within, RSI(5)<30 → entry on QQQ. VRP < 2 → GLD."""
    v = state.get('_vrp_override', vrp.get(date, None))
    if v is None or not np.isfinite(v):
        return {'allocations': state.get('current_alloc', {'GLD': 1.0})}

    rsi = qqq_rsi5.get(date, None)
    if rsi is None or not np.isfinite(rsi):
        return {'allocations': state.get('current_alloc', {'GLD': 1.0})}

    current = state.get('current_alloc', {'GLD': 1.0})

    if v < 2:
        alloc = {'GLD': 1.0}
        state['in_trade'] = False
    elif v > 5:
        # VRP regime is favorable
        if rsi < 30:
            alloc = {'QQQ': 1.0}
            state['in_trade'] = True
            state['entry_date'] = date
        elif state.get('in_trade', False):
            # Exit after 10 days or RSI > 60
            entry = state.get('entry_date', date)
            dates_held = vrp.index[(vrp.index > entry) & (vrp.index <= date)]
            if len(dates_held) >= 10 or rsi > 60:
                alloc = {'GLD': 1.0}
                state['in_trade'] = False
            else:
                alloc = {'QQQ': 1.0}
        else:
            alloc = current
    else:
        # VRP 2-5: hold position
        if state.get('in_trade', False):
            entry = state.get('entry_date', date)
            dates_held = vrp.index[(vrp.index > entry) & (vrp.index <= date)]
            if len(dates_held) >= 10 or (rsi is not None and rsi > 60):
                alloc = {'GLD': 1.0}
                state['in_trade'] = False
            else:
                alloc = {'QQQ': 1.0}
        else:
            alloc = current

    state['current_alloc'] = alloc
    return {'allocations': alloc}


# ── Run All Variants ─────────────────────────────────────────────────────
variants = {
    'A_VRP_Timing_QQQ': variant_a_signal,
    'B_VRP_Momentum_Blend': variant_b_signal,
    'C_VRP_Percentile': variant_c_signal,
    'D_VRP_Mean_Reversion': variant_d_signal,
    'E_VRP_Scaled_Sizing': variant_e_signal,
    'F_VRP_RSI_Combo': variant_f_signal,
}

# Buy-and-hold benchmarks
def bh_qqq_signal(date, state):
    return {'allocations': {'QQQ': 1.0}}

def bh_spy_signal(date, state):
    return {'allocations': {'SPY': 1.0}}

benchmarks = {
    'BH_QQQ': bh_qqq_signal,
    'BH_SPY': bh_spy_signal,
}

all_results = {}

# Run benchmarks first
print("\n=== Benchmarks ===")
for name, func in benchmarks.items():
    result = run_backtest(name, func)
    metrics = compute_metrics(result)
    if metrics:
        print(f"  {name}: Return={metrics['total_return_pct']:.1f}%, "
              f"Sharpe={metrics['sharpe']:.3f}, MaxDD={metrics['max_drawdown_pct']:.1f}%")
        all_results[name] = {'metrics': metrics}

# Run variants
print("\n=== Running Variants ===")
results_json = {}

for name, func in variants.items():
    print(f"\n--- {name} ---")

    # Real backtest
    result = run_backtest(name, func)
    if result is None:
        print(f"  SKIPPED: No data")
        continue

    metrics = compute_metrics(result)
    if metrics is None:
        print(f"  SKIPPED: Insufficient data for metrics")
        continue

    regime = compute_regime_metrics(result, spy_regime)

    print(f"  Return: {metrics['total_return_pct']:.1f}%, Sharpe: {metrics['sharpe']:.3f}, "
          f"Sortino: {metrics['sortino']:.3f}")
    print(f"  MaxDD: {metrics['max_drawdown_pct']:.1f}%, WR: {metrics['win_rate_pct']:.1f}%, "
          f"PF: {metrics['profit_factor']:.3f}")
    print(f"  Trades: {metrics['n_trades']}, Final Equity: ${metrics['final_equity']:.2f}")
    if regime:
        print(f"  Bull Sharpe: {regime['bull_sharpe']:.3f}, Bear Sharpe: {regime['bear_sharpe']:.3f}, "
              f"Gap: {regime['regime_gap']:.3f}")

    # Permutation test
    print(f"  Running {N_PERM} permutations ...")
    perm_p, n_valid = permutation_test(func, N_PERM)
    print(f"  Permutation p-value: {perm_p:.4f} ({n_valid} valid perms)")

    # 5-gate check
    gates = check_gates(metrics, regime, perm_p, metrics['n_trades'])
    gate_str = " | ".join([f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()])
    print(f"  Gates: {gate_str}")

    results_json[name] = {
        'metrics': {
            'total_return_pct': metrics['total_return_pct'],
            'ann_return_pct': metrics['ann_return_pct'],
            'ann_volatility_pct': metrics['ann_volatility_pct'],
            'sharpe': metrics['sharpe'],
            'sortino': metrics['sortino'],
            'max_drawdown_pct': metrics['max_drawdown_pct'],
            'profit_factor': metrics['profit_factor'],
            'win_rate_pct': metrics['win_rate_pct'],
            'n_trades': metrics['n_trades'],
            'final_equity': metrics['final_equity'],
        },
        'regime': {
            'bull_sharpe': regime['bull_sharpe'],
            'bear_sharpe': regime['bear_sharpe'],
            'regime_gap': regime['regime_gap'],
            'bull_days': regime['bull_days'],
            'bear_days': regime['bear_days'],
        },
        'permutation': {
            'p_value': round(perm_p, 4),
            'n_permutations': N_PERM,
            'n_valid': n_valid,
        },
        'gates': {k: v for k, v in gates.items()},
    }

# ── Summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("VOLATILITY RISK PREMIUM HARVESTING — RESULTS SUMMARY")
print("=" * 80)
print(f"{'Variant':<25} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} "
      f"{'WR%':>5} {'PF':>6} {'Trades':>7} {'Perm-p':>7} {'5-Gate':>7}")
print("-" * 95)

for name in sorted(results_json.keys()):
    r = results_json[name]
    m = r['metrics']
    p = r['permutation']['p_value']
    g = 'PASS' if r['gates']['all_pass'] else 'FAIL'
    print(f"{name:<25} {m['total_return_pct']:>7.1f}% {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
          f"{m['max_drawdown_pct']:>6.1f}% {m['win_rate_pct']:>5.1f} {m['profit_factor']:>6.3f} "
          f"{m['n_trades']:>7} {p:>7.4f} {g:>7}")

# Benchmarks
if 'BH_QQQ' in all_results:
    m = all_results['BH_QQQ']['metrics']
    print(f"{'BH_QQQ (benchmark)':<25} {m['total_return_pct']:>7.1f}% {m['sharpe']:>7.3f} "
          f"{m['sortino']:>8.3f} {m['max_drawdown_pct']:>6.1f}% {m['win_rate_pct']:>5.1f} "
          f"{m['profit_factor']:>6.3f} {'N/A':>7} {'N/A':>7} {'N/A':>7}")

# ── Save results ─────────────────────────────────────────────────────────
output = {
    'strategy': 'Volatility Risk Premium Harvesting',
    'thesis': 'VIX systematically overestimates realized vol by ~3-4 pts. Use VRP signal to time equity entries.',
    'backtest_period': f"{OOT_START} to {END}",
    'capital': CAPITAL,
    'slippage_pct': SLIPPAGE_PCT,
    'commission': 0,
    'vrp_stats': {
        'mean': round(float(vrp.mean()), 2),
        'median': round(float(vrp.median()), 2),
        'std': round(float(vrp.std()), 2),
        'min': round(float(vrp.min()), 2),
        'max': round(float(vrp.max()), 2),
    },
    'benchmarks': {k: v['metrics'] for k, v in all_results.items()},
    'variants': results_json,
    'generated_at': datetime.now().isoformat(),
}

out_path = Path("/home/jupiter/Lvl3Quant/data/vol_risk_premium_harvest_results.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {out_path}")

# Count passes
n_pass = sum(1 for v in results_json.values() if v['gates']['all_pass'])
print(f"\n{n_pass}/{len(results_json)} variants passed all 5 gates.")
