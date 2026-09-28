#!/usr/bin/env python3
"""
Defensive Value Backtest — 6 regime-aware strategy variants
Focus: strategies that work in BOTH bull and bear regimes (low regime gap)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

# ─── CONFIG ───────────────────────────────────────────────────────────────────
TICKERS = ['XLU', 'XLP', 'XLV', 'GLD', 'TLT', 'SHY', 'VNQ', 'SPY', 'QQQ', 'XLE']
DEFENSIVE_TICKERS = ['XLU', 'XLP', 'XLV', 'GLD', 'TLT']
START_DATE = '2020-01-01'  # need lookback before OOT
OOT_START = '2022-01-01'
OOT_END = '2026-07-28'
STARTING_CAPITAL = 645.0
N_PERMUTATIONS = 1000
np.random.seed(42)

# ─── DATA DOWNLOAD ────────────────────────────────────────────────────────────
print("Downloading price data...")
data = yf.download(TICKERS + ['^VIX'], start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    close = data['Close']
else:
    close = data

# Flatten any remaining multi-index
if isinstance(close.columns, pd.MultiIndex):
    close.columns = close.columns.get_level_values(-1)

# Separate VIX
vix = close['^VIX'].copy() if '^VIX' in close.columns else None
close = close[[t for t in TICKERS if t in close.columns]].copy()
close = close.ffill().dropna()

if vix is not None:
    vix = vix.reindex(close.index).ffill()

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")
print(f"Tickers available: {list(close.columns)}")

# ─── REGIME CLASSIFICATION ────────────────────────────────────────────────────
spy_sma200 = close['SPY'].rolling(200).mean()
regime = (close['SPY'] > spy_sma200).astype(int)  # 1=bull, 0=bear
regime = regime.reindex(close.index).ffill()

# Daily returns
returns = close.pct_change().dropna()

# OOT filter
oot_mask = returns.index >= OOT_START
oot_returns = returns[oot_mask]
oot_close = close.reindex(oot_returns.index)
oot_regime = regime.reindex(oot_returns.index)
if vix is not None:
    oot_vix = vix.reindex(oot_returns.index)

print(f"OOT period: {oot_returns.index[0].date()} to {oot_returns.index[-1].date()}, {len(oot_returns)} days")
bull_days = (oot_regime == 1).sum()
bear_days = (oot_regime == 0).sum()
print(f"Regime split: {bull_days} bull days, {bear_days} bear days")


# ─── HELPER FUNCTIONS ─────────────────────────────────────────────────────────
def compute_metrics(daily_rets, regime_series, name):
    """Compute all required metrics from a daily return series."""
    daily_rets = daily_rets.dropna()
    if len(daily_rets) < 20:
        return None

    total_ret = (1 + daily_rets).prod() - 1
    n_years = len(daily_rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = (daily_rets.mean() * 252) / max(ann_vol, 1e-8)

    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino = (daily_rets.mean() * 252) / max(downside, 1e-8)

    cum = (1 + daily_rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Monthly returns for win rate and PF
    monthly = daily_rets.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    monthly = monthly.dropna()
    win_rate = (monthly > 0).sum() / max(len(monthly), 1)
    gross_profit = monthly[monthly > 0].sum()
    gross_loss = abs(monthly[monthly < 0].sum())
    profit_factor = gross_profit / max(gross_loss, 1e-8)
    n_trades = len(monthly)  # monthly rebalances = trades

    # Regime-stratified Sharpe
    aligned_regime = regime_series.reindex(daily_rets.index).ffill()
    bull_rets = daily_rets[aligned_regime == 1]
    bear_rets = daily_rets[aligned_regime == 0]

    bull_sharpe = (bull_rets.mean() * 252) / max(bull_rets.std() * np.sqrt(252), 1e-8) if len(bull_rets) > 20 else 0
    bear_sharpe = (bear_rets.mean() * 252) / max(bear_rets.std() * np.sqrt(252), 1e-8) if len(bear_rets) > 20 else 0

    denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    return {
        'name': name,
        'total_return': round(total_ret * 100, 2),
        'cagr': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown': round(max_dd * 100, 2),
        'win_rate_monthly': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 3),
        'total_trades': n_trades,
        'bull_sharpe': round(bull_sharpe, 3),
        'bear_sharpe': round(bear_sharpe, 3),
        'regime_gap': round(regime_gap, 3),
        'daily_returns': daily_rets,  # keep for permutation test
    }


def permutation_test(actual_sharpe, daily_rets_dict, n_perms=N_PERMUTATIONS):
    """
    Shuffle which assets get which weights each month.
    Returns p-value = fraction of random shuffles beating actual Sharpe.
    """
    # Get all available monthly return series
    all_monthly = {}
    for ticker in TICKERS:
        if ticker in returns.columns:
            tr = returns[ticker][oot_mask]
            all_monthly[ticker] = tr

    count_better = 0
    for _ in range(n_perms):
        # Random weights each month
        monthly_dates = pd.date_range(OOT_START, OOT_END, freq='MS')
        sim_rets = []
        for mstart in monthly_dates:
            mend = mstart + pd.offsets.MonthEnd(1)
            # Random portfolio: pick 3 random ETFs, equal weight
            chosen = np.random.choice(TICKERS, size=min(3, len(TICKERS)), replace=False)
            period_rets = returns.loc[mstart:mend, [c for c in chosen if c in returns.columns]]
            if len(period_rets) > 0:
                avg_ret = period_rets.mean(axis=1)
                sim_rets.append(avg_ret)
        if sim_rets:
            combined = pd.concat(sim_rets)
            sim_sharpe = combined.mean() * 252 / max(combined.std() * np.sqrt(252), 1e-8)
            if sim_sharpe >= actual_sharpe:
                count_better += 1

    return count_better / n_perms


# ─── STRATEGY A: RISK PARITY ROTATION ─────────────────────────────────────────
print("\n[A] Risk Parity Rotation...")
def run_risk_parity():
    weights_history = []
    strat_rets = []
    rebal_dates = pd.date_range(OOT_START, OOT_END, freq='MS')

    for i, mstart in enumerate(rebal_dates):
        mend = mstart + pd.offsets.MonthEnd(1)
        # 20-day realized vol from prior data
        lookback_start = mstart - pd.Timedelta(days=40)
        lb_rets = returns.loc[lookback_start:mstart]
        if len(lb_rets) < 10:
            continue

        vols = lb_rets.std()
        inv_vol = 1.0 / vols.replace(0, np.inf)
        inv_vol = inv_vol.replace([np.inf, -np.inf], 0)
        w = inv_vol / inv_vol.sum()

        period_rets = oot_returns.loc[mstart:mend]
        if len(period_rets) > 0:
            port_ret = (period_rets * w).sum(axis=1)
            strat_rets.append(port_ret)

    if strat_rets:
        return pd.concat(strat_rets)
    return pd.Series(dtype=float)

strat_a_rets = run_risk_parity()
result_a = compute_metrics(strat_a_rets, oot_regime, "A_Risk_Parity_Rotation")


# ─── STRATEGY B: BEAR ALPHA ──────────────────────────────────────────────────
print("[B] Bear Alpha...")
def run_bear_alpha():
    strat_rets = []
    rebal_dates = pd.date_range(OOT_START, OOT_END, freq='MS')

    for mstart in rebal_dates:
        mend = mstart + pd.offsets.MonthEnd(1)
        # Check regime at start of month
        reg_at_start = regime.loc[:mstart].iloc[-1] if len(regime.loc[:mstart]) > 0 else 1

        period_rets = oot_returns.loc[mstart:mend]
        if len(period_rets) == 0:
            continue

        if reg_at_start == 0:  # bear
            # Equal weight XLU + XLP + GLD
            bear_tickers = [t for t in ['XLU', 'XLP', 'GLD'] if t in period_rets.columns]
            if bear_tickers:
                port_ret = period_rets[bear_tickers].mean(axis=1)
                strat_rets.append(port_ret)
        else:  # bull
            if 'QQQ' in period_rets.columns:
                strat_rets.append(period_rets['QQQ'])

    if strat_rets:
        return pd.concat(strat_rets)
    return pd.Series(dtype=float)

strat_b_rets = run_bear_alpha()
result_b = compute_metrics(strat_b_rets, oot_regime, "B_Bear_Alpha")


# ─── STRATEGY C: MINIMUM VARIANCE PORTFOLIO ──────────────────────────────────
print("[C] Minimum Variance Portfolio...")
def run_min_variance():
    strat_rets = []
    rebal_dates = pd.date_range(OOT_START, OOT_END, freq='MS')
    available = [t for t in TICKERS if t in returns.columns]
    n = len(available)

    for mstart in rebal_dates:
        mend = mstart + pd.offsets.MonthEnd(1)
        lookback_start = mstart - pd.Timedelta(days=90)
        lb_rets = returns.loc[lookback_start:mstart, available]
        if len(lb_rets) < 30:
            continue

        cov = lb_rets.cov().values * 252  # annualize

        # Min variance optimization (long-only)
        def port_var(w):
            return w @ cov @ w

        n_assets = len(available)
        x0 = np.ones(n_assets) / n_assets
        bounds = [(0, 1)] * n_assets
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

        try:
            res = minimize(port_var, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            w = res.x if res.success else x0
        except:
            w = x0

        period_rets = oot_returns.loc[mstart:mend, available]
        if len(period_rets) > 0:
            port_ret = (period_rets.values @ w)
            strat_rets.append(pd.Series(port_ret, index=period_rets.index))

    if strat_rets:
        return pd.concat(strat_rets)
    return pd.Series(dtype=float)

strat_c_rets = run_min_variance()
result_c = compute_metrics(strat_c_rets, oot_regime, "C_Min_Variance")


# ─── STRATEGY D: DEFENSIVE MOMENTUM ──────────────────────────────────────────
print("[D] Defensive Momentum...")
def run_defensive_momentum():
    strat_rets = []
    rebal_dates = pd.date_range(OOT_START, OOT_END, freq='MS')

    for mstart in rebal_dates:
        mend = mstart + pd.offsets.MonthEnd(1)
        lookback_start = mstart - pd.Timedelta(days=95)

        reg_at_start = regime.loc[:mstart].iloc[-1] if len(regime.loc[:mstart]) > 0 else 1

        # Determine eligible universe
        if reg_at_start == 0:  # bear → defensive only
            eligible = [t for t in DEFENSIVE_TICKERS if t in close.columns]
        else:  # bull → all
            eligible = [t for t in TICKERS if t in close.columns]

        # 3-month momentum
        lb_close = close.loc[lookback_start:mstart, eligible]
        if len(lb_close) < 20:
            continue
        momentum = lb_close.iloc[-1] / lb_close.iloc[0] - 1
        top3 = momentum.nlargest(3).index.tolist()

        period_rets = oot_returns.loc[mstart:mend, top3]
        if len(period_rets) > 0:
            port_ret = period_rets.mean(axis=1)
            strat_rets.append(port_ret)

    if strat_rets:
        return pd.concat(strat_rets)
    return pd.Series(dtype=float)

strat_d_rets = run_defensive_momentum()
result_d = compute_metrics(strat_d_rets, oot_regime, "D_Defensive_Momentum")


# ─── STRATEGY E: TAIL HEDGE OVERLAY ──────────────────────────────────────────
print("[E] Tail Hedge Overlay...")
def run_tail_hedge():
    if vix is None:
        print("  WARNING: VIX data not available, skipping")
        return pd.Series(dtype=float)

    strat_rets = []
    for date in oot_returns.index:
        v = oot_vix.get(date, None)
        if v is None or pd.isna(v):
            v = 20  # default

        qqq_ret = oot_returns.loc[date, 'QQQ'] if 'QQQ' in oot_returns.columns else 0
        gld_ret = oot_returns.loc[date, 'GLD'] if 'GLD' in oot_returns.columns else 0
        tlt_ret = oot_returns.loc[date, 'TLT'] if 'TLT' in oot_returns.columns else 0

        if v > 30:
            # 100% defensive: 50% GLD + 50% TLT
            port_ret = 0.5 * gld_ret + 0.5 * tlt_ret
        elif v > 22:
            # 50% QQQ + 50% GLD
            port_ret = 0.5 * qqq_ret + 0.5 * gld_ret
        else:
            # 100% QQQ
            port_ret = qqq_ret

        strat_rets.append(port_ret)

    return pd.Series(strat_rets, index=oot_returns.index)

strat_e_rets = run_tail_hedge()
result_e = compute_metrics(strat_e_rets, oot_regime, "E_Tail_Hedge_Overlay")


# ─── STRATEGY F: SECTOR ROTATION + CASH ──────────────────────────────────────
print("[F] Sector Rotation + Cash...")
def run_sector_rotation_cash():
    strat_rets = []
    rebal_dates = pd.date_range(OOT_START, OOT_END, freq='MS')
    sector_tickers = [t for t in ['XLU', 'XLP', 'XLV', 'XLE', 'VNQ', 'QQQ', 'SPY'] if t in close.columns]

    for mstart in rebal_dates:
        mend = mstart + pd.offsets.MonthEnd(1)
        lookback_start = mstart - pd.Timedelta(days=90)

        lb_close = close.loc[lookback_start:mstart, sector_tickers]
        if len(lb_close) < 20:
            continue

        # 60-day momentum
        momentum = lb_close.iloc[-1] / lb_close.iloc[0] - 1

        # Filter positive momentum only
        pos_mom = momentum[momentum > 0]

        period_rets = oot_returns.loc[mstart:mend]
        if len(period_rets) == 0:
            continue

        if len(pos_mom) == 0:
            # All negative momentum → 100% SHY (cash)
            if 'SHY' in period_rets.columns:
                strat_rets.append(period_rets['SHY'])
        else:
            # Top 2 by momentum
            top2 = pos_mom.nlargest(2).index.tolist()
            port_ret = period_rets[top2].mean(axis=1)
            strat_rets.append(port_ret)

    if strat_rets:
        return pd.concat(strat_rets)
    return pd.Series(dtype=float)

strat_f_rets = run_sector_rotation_cash()
result_f = compute_metrics(strat_f_rets, oot_regime, "F_Sector_Rotation_Cash")


# ─── PERMUTATION TESTS ───────────────────────────────────────────────────────
print("\nRunning permutation tests (1000 shuffles each)...")
results_all = []
for label, result in [('A', result_a), ('B', result_b), ('C', result_c),
                       ('D', result_d), ('E', result_e), ('F', result_f)]:
    if result is None:
        print(f"  [{label}] Skipped (insufficient data)")
        continue

    daily_rets = result.pop('daily_returns')
    perm_p = permutation_test(result['sharpe'], daily_rets)
    result['perm_p'] = round(perm_p, 4)

    # 5-gate validation
    gates = {
        'sharpe_gt_0.5': result['sharpe'] > 0.5,
        'perm_p_lt_0.05': perm_p < 0.05,
        'regime_gap_lt_0.5': result['regime_gap'] < 0.5,
        'mdd_gt_neg50': result['max_drawdown'] > -50,
        'trades_gte_20': result['total_trades'] >= 20,
    }
    result['gates'] = gates
    result['gates_passed'] = sum(gates.values())
    result['all_gates_pass'] = all(gates.values())

    # Final equity
    cum_ret = (1 + daily_rets).prod()
    result['final_equity'] = round(STARTING_CAPITAL * cum_ret, 2)

    results_all.append(result)
    status = "PASS" if result['all_gates_pass'] else "FAIL"
    print(f"  [{label}] {result['name']}: Sharpe={result['sharpe']:.3f}, "
          f"RegimeGap={result['regime_gap']:.3f}, perm_p={perm_p:.4f} → {status} ({result['gates_passed']}/5)")


# ─── SUMMARY ──────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("DEFENSIVE VALUE BACKTEST RESULTS — OOT Jan 2022 - Jul 2026")
print("="*80)
print(f"{'Strategy':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} {'MDD%':>7} {'RGap':>6} {'perm_p':>7} {'Gates':>6}")
print("-"*80)
for r in sorted(results_all, key=lambda x: -x['sharpe']):
    status = "PASS" if r['all_gates_pass'] else f"{r['gates_passed']}/5"
    print(f"{r['name']:<30} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['cagr']:>6.1f}% {r['max_drawdown']:>6.1f}% "
          f"{r['regime_gap']:>6.3f} {r['perm_p']:>7.4f} {status:>6}")

print("\nRegime-Stratified Detail:")
print(f"{'Strategy':<30} {'Bull Sharpe':>12} {'Bear Sharpe':>12} {'Regime Gap':>11}")
print("-"*70)
for r in sorted(results_all, key=lambda x: x['regime_gap']):
    print(f"{r['name']:<30} {r['bull_sharpe']:>12.3f} {r['bear_sharpe']:>12.3f} {r['regime_gap']:>11.3f}")


# ─── SAVE RESULTS ─────────────────────────────────────────────────────────────
output = {
    'run_timestamp': datetime.now().isoformat(),
    'config': {
        'oot_start': OOT_START,
        'oot_end': OOT_END,
        'starting_capital': STARTING_CAPITAL,
        'tickers': TICKERS,
        'n_permutations': N_PERMUTATIONS,
        'regime': 'SPY_200SMA',
        'bull_days': int(bull_days),
        'bear_days': int(bear_days),
    },
    'strategies': results_all,
    'summary': {
        'total_strategies': len(results_all),
        'passing_all_gates': sum(1 for r in results_all if r['all_gates_pass']),
        'best_regime_gap': min(r['regime_gap'] for r in results_all) if results_all else None,
        'best_sharpe': max(r['sharpe'] for r in results_all) if results_all else None,
        'strategies_with_low_regime_gap': [r['name'] for r in results_all if r['regime_gap'] < 0.5],
    }
}

output_path = '/home/jupiter/Lvl3Quant/data/defensive_value_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print(f"\nStrategies passing ALL 5 gates: {output['summary']['passing_all_gates']}/{len(results_all)}")
print(f"Lowest regime gap: {output['summary']['best_regime_gap']}")
