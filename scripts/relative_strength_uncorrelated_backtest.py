#!/usr/bin/env python3
"""
Relative Strength Uncorrelated Assets Backtest
Tests 6 variants that rotate into low-correlation assets based on relative strength signals.
"""

import json
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = ['GLD', 'TLT', 'UUP', 'SPY', 'QQQ', 'SHV']
START = '2022-01-01'
END = '2026-07-29'
INITIAL_CAPITAL = 645.0
SLIPPAGE = 0.0002  # 0.02% per rebalance trade
N_PERMS = 1000
SEED = 42

# ── Download Data ───────────────────────────────────────────────────────
print("Downloading data...")
data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
prices = data['Close'].dropna()
returns = prices.pct_change().dropna()
print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")

ASSETS = ['GLD', 'TLT', 'UUP']

# ── Helpers ─────────────────────────────────────────────────────────────
def compute_equity(returns_series, initial=INITIAL_CAPITAL):
    return initial * (1 + returns_series).cumprod()

def apply_slippage(daily_returns, rebalance_mask):
    """Apply slippage on rebalance days."""
    adj = daily_returns.copy()
    common = rebalance_mask.reindex(adj.index).fillna(False)
    adj[common] -= SLIPPAGE
    return adj

def sharpe(returns_series):
    if returns_series.std() == 0:
        return 0.0
    return returns_series.mean() / returns_series.std() * np.sqrt(252)

def max_drawdown(equity):
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return dd.min()

def regime_gap(strat_returns, spy_returns):
    """Sharpe in up-SPY days vs down-SPY days."""
    up = spy_returns >= 0
    down = spy_returns < 0
    s_up = sharpe(strat_returns[up]) if up.sum() > 10 else 0
    s_down = sharpe(strat_returns[down]) if down.sum() > 10 else 0
    denom = max(abs(s_up), abs(s_down), 1e-9)
    return abs(s_up - s_down) / denom

def permutation_test(actual_sharpe, daily_returns, selections, n_perms=N_PERMS):
    """Shuffle which asset is selected each period, recompute Sharpe. Vectorized."""
    rng = np.random.RandomState(SEED)
    sel_clean = selections.dropna()
    unique_sels = sel_clean.unique()
    if len(unique_sels) <= 1:
        return 1.0

    # Identify block boundaries (where selection changes)
    changes = sel_clean.ne(sel_clean.shift())
    block_ids = changes.cumsum() - 1
    n_blocks = block_ids.max() + 1

    # Build returns matrix for each asset, aligned to selection index
    common_idx = sel_clean.index.intersection(returns.index)
    ret_matrix = returns.loc[common_idx, [a for a in unique_sels if a in returns.columns]]
    block_ids_aligned = block_ids.reindex(common_idx)

    count_better = 0
    for _ in range(n_perms):
        # Random asset for each block
        perm_choices = rng.choice([a for a in unique_sels if a in returns.columns], size=int(n_blocks))
        # Map block_id -> asset -> return
        perm_assets = pd.Series(perm_choices[block_ids_aligned.values.astype(int)], index=common_idx)
        perm_ret = pd.Series(0.0, index=common_idx)
        for asset in ret_matrix.columns:
            mask = perm_assets == asset
            perm_ret[mask] = ret_matrix.loc[mask, asset].values
        if sharpe(perm_ret) >= actual_sharpe:
            count_better += 1
    return count_better / n_perms

def qqq_correlation(strat_returns):
    qqq_ret = returns['QQQ'].reindex(strat_returns.index)
    common = pd.concat([strat_returns, qqq_ret], axis=1).dropna()
    if len(common) < 20:
        return np.nan
    return common.iloc[:, 0].corr(common.iloc[:, 1])

def count_trades(selections):
    """Count number of rebalance trades (asset switches)."""
    changes = selections.ne(selections.shift())
    return changes.sum()

def get_weekly_dates(index):
    """Get weekly rebalance dates (every Friday, or last trading day of week)."""
    weekly = index.to_series().groupby(index.to_period('W')).last()
    return weekly.values

def get_monthly_dates(index):
    """Get monthly rebalance dates."""
    monthly = index.to_series().groupby(index.to_period('M')).last()
    return monthly.values


# ── Strategy A: Safe Haven Relative Strength ────────────────────────────
def strategy_a():
    """Among GLD, TLT, UUP, pick best 20-day relative strength vs SPY. Weekly rebalance."""
    rel_strength = pd.DataFrame(index=prices.index)
    for a in ASSETS:
        rel_strength[a] = (prices[a] / prices['SPY']).pct_change(20)

    weekly_dates = get_weekly_dates(prices.index)
    selection = pd.Series(index=prices.index, dtype=str)

    current_asset = None
    for dt in prices.index:
        if dt in weekly_dates:
            row = rel_strength.loc[dt].dropna()
            if len(row) == len(ASSETS):
                current_asset = row.idxmax()
        if current_asset is not None:
            selection.loc[dt] = current_asset

    selection = selection.dropna()
    strat_ret = pd.Series(0.0, index=returns.index)
    for dt in strat_ret.index:
        if dt in selection.index and selection.loc[dt] in returns.columns:
            strat_ret.loc[dt] = returns.loc[dt, selection.loc[dt]]

    rebal_mask = selection.ne(selection.shift())
    strat_ret = apply_slippage(strat_ret, rebal_mask)
    return strat_ret[strat_ret.index >= selection.index[0]], selection


# ── Strategy B: Commodity vs Bond Rotation ──────────────────────────────
def strategy_b():
    """Rotate between GLD, TLT, UUP based on 10-day momentum. Weekly rebalance."""
    mom = pd.DataFrame(index=prices.index)
    for a in ASSETS:
        mom[a] = prices[a].pct_change(10)

    weekly_dates = get_weekly_dates(prices.index)
    selection = pd.Series(index=prices.index, dtype=str)

    current_asset = None
    for dt in prices.index:
        if dt in weekly_dates:
            row = mom.loc[dt].dropna()
            if len(row) == len(ASSETS):
                current_asset = row.idxmax()
        if current_asset is not None:
            selection.loc[dt] = current_asset

    selection = selection.dropna()
    strat_ret = pd.Series(0.0, index=returns.index)
    for dt in strat_ret.index:
        if dt in selection.index and selection.loc[dt] in returns.columns:
            strat_ret.loc[dt] = returns.loc[dt, selection.loc[dt]]

    rebal_mask = selection.ne(selection.shift())
    strat_ret = apply_slippage(strat_ret, rebal_mask)
    return strat_ret[strat_ret.index >= selection.index[0]], selection


# ── Strategy C: Anti-Correlation Filter ─────────────────────────────────
def strategy_c():
    """Hold whichever of GLD, TLT, UUP has most NEGATIVE 20-day rolling correlation with QQQ. Monthly."""
    roll_corr = pd.DataFrame(index=returns.index)
    for a in ASSETS:
        roll_corr[a] = returns[a].rolling(20).corr(returns['QQQ'])

    monthly_dates = get_monthly_dates(prices.index)
    selection = pd.Series(index=returns.index, dtype=str)

    current_asset = None
    for dt in returns.index:
        if dt in monthly_dates:
            row = roll_corr.loc[dt].dropna()
            if len(row) == len(ASSETS):
                current_asset = row.idxmin()  # most negative correlation
        if current_asset is not None:
            selection.loc[dt] = current_asset

    selection = selection.dropna()
    strat_ret = pd.Series(0.0, index=returns.index)
    for dt in strat_ret.index:
        if dt in selection.index and selection.loc[dt] in returns.columns:
            strat_ret.loc[dt] = returns.loc[dt, selection.loc[dt]]

    rebal_mask = selection.ne(selection.shift())
    strat_ret = apply_slippage(strat_ret, rebal_mask)
    return strat_ret[strat_ret.index >= selection.index[0]], selection


# ── Strategy D: Volatility-Adjusted Relative Strength ───────────────────
def strategy_d():
    """Best risk-adjusted momentum (20d return / 20d vol). Weekly. Vol-target 8%."""
    risk_adj = pd.DataFrame(index=prices.index)
    vol = pd.DataFrame(index=returns.index)
    for a in ASSETS:
        ret20 = prices[a].pct_change(20)
        vol20 = returns[a].rolling(20).std() * np.sqrt(252)
        risk_adj[a] = ret20 / vol20.reindex(ret20.index).replace(0, np.nan)
        vol[a] = vol20

    weekly_dates = get_weekly_dates(prices.index)
    selection = pd.Series(index=prices.index, dtype=str)
    vol_scalar = pd.Series(1.0, index=prices.index)
    TARGET_VOL = 0.08

    current_asset = None
    current_scalar = 1.0
    for dt in prices.index:
        if dt in weekly_dates:
            row = risk_adj.loc[dt].dropna()
            if len(row) == len(ASSETS):
                current_asset = row.idxmax()
                asset_vol = vol.loc[dt, current_asset] if dt in vol.index and not np.isnan(vol.loc[dt, current_asset]) else TARGET_VOL
                current_scalar = min(TARGET_VOL / max(asset_vol, 0.01), 2.0)  # cap leverage at 2x
        if current_asset is not None:
            selection.loc[dt] = current_asset
            vol_scalar.loc[dt] = current_scalar

    selection = selection.dropna()
    strat_ret = pd.Series(0.0, index=returns.index)
    for dt in strat_ret.index:
        if dt in selection.index and selection.loc[dt] in returns.columns:
            strat_ret.loc[dt] = returns.loc[dt, selection.loc[dt]] * vol_scalar.loc[dt]

    rebal_mask = selection.ne(selection.shift())
    strat_ret = apply_slippage(strat_ret, rebal_mask)
    return strat_ret[strat_ret.index >= selection.index[0]], selection


# ── Strategy E: Dual Momentum ──────────────────────────────────────────
def strategy_e():
    """Hold best of GLD/TLT/UUP ONLY if absolute 20d return > 0. Otherwise cash. Weekly."""
    abs_ret = pd.DataFrame(index=prices.index)
    for a in ASSETS:
        abs_ret[a] = prices[a].pct_change(20)

    CASH_DAILY = 0.0002  # 0.02% daily for SHV proxy

    weekly_dates = get_weekly_dates(prices.index)
    selection = pd.Series(index=prices.index, dtype=str)

    current_asset = None
    for dt in prices.index:
        if dt in weekly_dates:
            row = abs_ret.loc[dt].dropna()
            if len(row) == len(ASSETS):
                best = row.idxmax()
                if row[best] > 0:
                    current_asset = best
                else:
                    current_asset = 'CASH'
        if current_asset is not None:
            selection.loc[dt] = current_asset

    selection = selection.dropna()
    strat_ret = pd.Series(0.0, index=returns.index)
    for dt in strat_ret.index:
        if dt in selection.index:
            asset = selection.loc[dt]
            if asset == 'CASH':
                strat_ret.loc[dt] = CASH_DAILY
            elif asset in returns.columns:
                strat_ret.loc[dt] = returns.loc[dt, asset]

    rebal_mask = selection.ne(selection.shift())
    strat_ret = apply_slippage(strat_ret, rebal_mask)
    return strat_ret[strat_ret.index >= selection.index[0]], selection


# ── Strategy F: Adaptive Hedge ──────────────────────────────────────────
def strategy_f():
    """Hold asset with most NEGATIVE 60-day beta to QQQ. Monthly rebalance."""
    betas = pd.DataFrame(index=returns.index)
    for a in ASSETS:
        cov = returns[a].rolling(60).cov(returns['QQQ'])
        var_qqq = returns['QQQ'].rolling(60).var()
        betas[a] = cov / var_qqq.replace(0, np.nan)

    monthly_dates = get_monthly_dates(prices.index)
    selection = pd.Series(index=returns.index, dtype=str)

    current_asset = None
    for dt in returns.index:
        if dt in monthly_dates:
            row = betas.loc[dt].dropna()
            if len(row) == len(ASSETS):
                current_asset = row.idxmin()  # most negative beta
        if current_asset is not None:
            selection.loc[dt] = current_asset

    selection = selection.dropna()
    strat_ret = pd.Series(0.0, index=returns.index)
    for dt in strat_ret.index:
        if dt in selection.index and selection.loc[dt] in returns.columns:
            strat_ret.loc[dt] = returns.loc[dt, selection.loc[dt]]

    rebal_mask = selection.ne(selection.shift())
    strat_ret = apply_slippage(strat_ret, rebal_mask)
    return strat_ret[strat_ret.index >= selection.index[0]], selection


# ── Run All Strategies ──────────────────────────────────────────────────
strategies = {
    'A_SafeHavenRS': strategy_a,
    'B_CommodityBondRotation': strategy_b,
    'C_AntiCorrelation': strategy_c,
    'D_VolAdjRS': strategy_d,
    'E_DualMomentum': strategy_e,
    'F_AdaptiveHedge': strategy_f,
}

spy_ret = returns['SPY']
results = {}

for name, func in strategies.items():
    print(f"\n{'='*60}")
    print(f"Strategy {name}")
    print('='*60)

    strat_ret, selection = func()

    # Align with SPY returns
    common_idx = strat_ret.index.intersection(spy_ret.index)
    strat_ret = strat_ret.loc[common_idx]
    spy_aligned = spy_ret.loc[common_idx]

    # Metrics
    s = sharpe(strat_ret)
    eq = compute_equity(strat_ret)
    mdd = max_drawdown(eq)
    n_trades = count_trades(selection)
    qqq_corr = qqq_correlation(strat_ret)
    rg = regime_gap(strat_ret, spy_aligned)

    # Permutation test
    print(f"  Running permutation test ({N_PERMS} shuffles)...")
    p_val = permutation_test(s, strat_ret, selection)

    # Annual return
    n_years = len(strat_ret) / 252
    total_ret = eq.iloc[-1] / INITIAL_CAPITAL - 1
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.1)) - 1

    # Sortino
    downside = strat_ret[strat_ret < 0]
    sortino = strat_ret.mean() / (downside.std() if len(downside) > 0 and downside.std() > 0 else 1e-9) * np.sqrt(252)

    # 5-Gate validation
    gate_sharpe = s > 0.5
    gate_perm = p_val < 0.05
    gate_regime = rg < 0.5
    gate_mdd = mdd > -0.50
    gate_trades = n_trades >= 20
    gates_passed = sum([gate_sharpe, gate_perm, gate_regime, gate_mdd, gate_trades])
    all_pass = gates_passed == 5

    print(f"  Sharpe: {s:.3f} | Sortino: {sortino:.3f} | Ann Return: {ann_ret:.1%}")
    print(f"  Max DD: {mdd:.1%} | Trades: {n_trades} | QQQ Corr: {qqq_corr:.3f}")
    print(f"  Perm p-val: {p_val:.3f} | Regime Gap: {rg:.3f}")
    print(f"  Final Equity: ${eq.iloc[-1]:.2f}")
    print(f"  Gates: Sharpe={'PASS' if gate_sharpe else 'FAIL'} | Perm={'PASS' if gate_perm else 'FAIL'} | "
          f"Regime={'PASS' if gate_regime else 'FAIL'} | MDD={'PASS' if gate_mdd else 'FAIL'} | "
          f"Trades={'PASS' if gate_trades else 'FAIL'}")
    print(f"  >> {'ALL GATES PASSED' if all_pass else f'{gates_passed}/5 gates passed'}")

    results[name] = {
        'sharpe': round(s, 4),
        'sortino': round(sortino, 4),
        'annual_return': round(ann_ret, 4),
        'total_return': round(total_ret, 4),
        'max_drawdown': round(mdd, 4),
        'n_trades': int(n_trades),
        'qqq_correlation': round(qqq_corr, 4),
        'perm_p_value': round(p_val, 4),
        'regime_gap': round(rg, 4),
        'final_equity': round(float(eq.iloc[-1]), 2),
        'gates': {
            'sharpe_gt_0.5': bool(gate_sharpe),
            'perm_p_lt_0.05': bool(gate_perm),
            'regime_gap_lt_0.5': bool(gate_regime),
            'max_dd_gt_neg50': bool(gate_mdd),
            'trades_gte_20': bool(gate_trades),
        },
        'gates_passed': int(gates_passed),
        'all_gates_passed': bool(all_pass),
    }

# ── Save Results ────────────────────────────────────────────────────────
output = {
    'metadata': {
        'run_date': datetime.now().isoformat(),
        'start_date': START,
        'end_date': END,
        'initial_capital': INITIAL_CAPITAL,
        'slippage_pct': SLIPPAGE * 100,
        'n_permutations': N_PERMS,
        'assets': ASSETS,
    },
    'strategies': results,
}

output_path = '/home/jupiter/Lvl3Quant/data/relative_strength_uncorrelated_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to {output_path}")
print(f"{'='*60}")

# Summary
print("\n\nSUMMARY")
print(f"{'Strategy':<30} {'Sharpe':>7} {'QQQ Corr':>9} {'Gates':>6} {'Pass?':>6}")
print("-" * 62)
for name, r in results.items():
    print(f"{name:<30} {r['sharpe']:>7.3f} {r['qqq_correlation']:>9.3f} {r['gates_passed']:>4}/5  {'YES' if r['all_gates_passed'] else 'NO':>5}")
