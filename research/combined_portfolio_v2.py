"""
Combined Portfolio Backtest v2 — HC #659 (permutation test) + HC #428 R1 (regime symmetry)

Permutation test design (HC #659):
  - Null: random convex combination weights across the 4 strategies
  - Test statistic: Sharpe of the actual combination vs distribution under random weights
  - 1000 random weight draws, each yielding a Sharpe
  - p-value = fraction of random combos >= actual combo Sharpe
  - This tests whether the specific weighting scheme (equal-weight, risk-parity)
    adds value over random allocation.
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTDIR = Path('/home/jupiter/Lvl3Quant/output/multi_strategy_portfolio_v2')
OUTDIR.mkdir(parents=True, exist_ok=True)

TRADING_DAYS_YR = 252
np.random.seed(42)

# ============================================================
# DATA LOADING
# ============================================================

existing = pd.read_parquet('/home/jupiter/Lvl3Quant/output/multi_strategy_portfolio/combined_equity.parquet')
existing['date'] = pd.to_datetime(existing['date'])
existing = existing.set_index('date').sort_index()

v5_nav = existing['v5_combined']
v5_ret = v5_nav.pct_change().dropna()
v3_nav = existing['etf_v3']
v3_ret = v3_nav.pct_change().dropna()

base = '/home/jupiter/Lvl3Quant/output/rotation_universe_exploration/'
v4_df = pd.read_csv(base + 'returns_Wide_Sector+Intl_K4.csv')
v4_df['date'] = pd.to_datetime(v4_df['date'])
v4_df = v4_df.set_index('date').sort_index()
v4_ret_raw = v4_df['return']

ma_df = pd.read_csv(base + 'returns_Asset_Class_Factor_K4.csv')
ma_df['date'] = pd.to_datetime(ma_df['date'])
ma_df = ma_df.set_index('date').sort_index()
ma_ret_raw = ma_df['return']

# Align
all_dates = (v5_ret.index
             .intersection(v3_ret.index)
             .intersection(v4_ret_raw.index)
             .intersection(ma_ret_raw.index)
             .sort_values())

v5 = v5_ret.reindex(all_dates).fillna(0.0)
v3 = v3_ret.reindex(all_dates).fillna(0.0)
v4 = v4_ret_raw.reindex(all_dates).fillna(0.0)
ma = ma_ret_raw.reindex(all_dates).fillna(0.0)
n_days = len(all_dates)
print(f"Data: {all_dates[0].date()} to {all_dates[-1].date()}, n={n_days}")

# SPY regime
import yfinance as yf
spy_raw = yf.download('SPY', start='2017-12-01', end='2026-03-15', auto_adjust=True, progress=False)
if isinstance(spy_raw.columns, pd.MultiIndex):
    spy_close = spy_raw['Close']['SPY']
else:
    spy_close = spy_raw['Close']
spy_close.index = pd.to_datetime(spy_close.index).normalize()
spy_close = spy_close.reindex(all_dates).ffill()
spy_ret_regime = spy_close.pct_change()
FLAT_THRESH = 0.002
def classify_regime(r):
    if pd.isna(r): return 'unknown'
    if r > FLAT_THRESH: return 'green'
    elif r < -FLAT_THRESH: return 'red'
    else: return 'flat'
regimes = spy_ret_regime.map(classify_regime).reindex(all_dates)
print(f"Regime counts: {regimes.value_counts().to_dict()}")

rets = pd.DataFrame({'V5_CSP': v5, 'ETF_v3': v3, 'ETF_v4_Wide': v4, 'Multi_Asset': ma})

# ============================================================
# METRICS
# ============================================================

def compute_sharpe_from_ret(r_array):
    """Fast Sharpe from numpy array."""
    if len(r_array) < 60:
        return 0.0
    n_yr = len(r_array) / TRADING_DAYS_YR
    cum = np.prod(1 + r_array) - 1
    cagr = (1 + cum) ** (1.0 / n_yr) - 1
    vol = r_array.std() * np.sqrt(TRADING_DAYS_YR)
    return cagr / vol if vol > 1e-8 else 0.0

def compute_metrics(ret_series, name=''):
    r = ret_series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 252:
        return None
    r_arr = r.values
    n_years = len(r_arr) / TRADING_DAYS_YR
    cum_ret = np.prod(1 + r_arr) - 1
    cagr = (1 + cum_ret) ** (1.0 / n_years) - 1
    ann_vol = r_arr.std() * np.sqrt(TRADING_DAYS_YR)
    sharpe = cagr / ann_vol if ann_vol > 1e-8 else 0.0
    downside = r_arr[r_arr < 0]
    dd_vol = downside.std() * np.sqrt(TRADING_DAYS_YR) if len(downside) > 0 else np.nan
    sortino = cagr / dd_vol if (dd_vol and dd_vol > 1e-8) else np.nan
    cumulative = np.cumprod(1 + r_arr)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / rolling_max
    maxdd = float(drawdown.min())
    calmar = cagr / abs(maxdd) if abs(maxdd) > 1e-8 else 0.0
    wr = float((r_arr > 0).mean())
    gross_wins = r_arr[r_arr > 0].sum()
    gross_loss = abs(r_arr[r_arr < 0].sum())
    pf = gross_wins / gross_loss if gross_loss > 1e-10 else float('inf')
    # Day concentration
    nav_curve = np.cumprod(1 + r_arr)
    nav_shifted = np.concatenate([[1.0], nav_curve[:-1]])
    daily_pnl_usd = r_arr * nav_shifted
    total_pnl = daily_pnl_usd.sum()
    top1_conc = float(daily_pnl_usd.max() / total_pnl) if total_pnl > 0 else np.nan
    return {
        'name': name, 'sharpe': float(sharpe), 'sortino': float(sortino) if not np.isnan(sortino) else None,
        'cagr': float(cagr), 'ann_vol': float(ann_vol),
        'maxdd': float(maxdd), 'calmar': float(calmar),
        'wr': float(wr), 'pf': float(pf), 'top1_conc': float(top1_conc) if not np.isnan(top1_conc) else None,
        'n_days': int(len(r_arr)), 'n_years': float(n_years),
    }

def regime_split_metrics(ret_series, regimes_series):
    out = {}
    for regime in ['green', 'red', 'flat']:
        mask = regimes_series.reindex(ret_series.index) == regime
        r_sub = ret_series[mask].dropna()
        if len(r_sub) < 30:
            out[f'{regime}_sharpe'] = None
            out[f'{regime}_n'] = int(mask.sum())
            continue
        ann_vol = r_sub.std() * np.sqrt(TRADING_DAYS_YR)
        ann_ret = r_sub.mean() * TRADING_DAYS_YR
        sharpe = ann_ret / ann_vol if ann_vol > 1e-8 else 0.0
        out[f'{regime}_sharpe'] = float(sharpe)
        out[f'{regime}_n'] = int(mask.sum())
    g = out.get('green_sharpe') or 0
    r = out.get('red_sharpe') or 0
    denom = max(abs(g), abs(r))
    out['regime_gap'] = float(abs(g - r) / denom) if denom > 1e-8 else None
    out['regime_pass'] = bool(out['regime_gap'] is not None and out['regime_gap'] <= 0.50)
    return out

# ============================================================
# INDIVIDUAL STRATEGY METRICS
# ============================================================
print("\n=== INDIVIDUAL STRATEGIES ===")
individual_metrics = {}
for name, r in [('V5_CSP', v5), ('ETF_v3', v3), ('ETF_v4_Wide', v4), ('Multi_Asset', ma)]:
    m = compute_metrics(r, name=name)
    rm = regime_split_metrics(r, regimes)
    m.update(rm)
    individual_metrics[name] = m
    print(f"  {name:15s}: Sharpe={m['sharpe']:.3f} Sortino={m['sortino']:.3f} "
          f"MaxDD={m['maxdd']:.1%} CAGR={m['cagr']:.1%} | "
          f"RegimeGap={rm['regime_gap']:.3f if rm['regime_gap'] else 'N/A'} "
          f"{'PASS' if rm['regime_pass'] else 'FAIL'}")

# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================

# (a) Equal weight 4-way
ew4_ret = rets.mean(axis=1)

# (b) Risk Parity 4-way (63-day rolling inverse-vol, lagged 1 day)
ROLL_WIN = 63
rolling_vol = rets.rolling(ROLL_WIN).std()
rolling_vol = rolling_vol.replace(0, np.nan).ffill()
inv_vol = 1.0 / rolling_vol
weights_rp = inv_vol.div(inv_vol.sum(axis=1), axis=0)
rp4_ret = (rets * weights_rp.shift(1)).sum(axis=1)
rp4_ret.iloc[:ROLL_WIN] = np.nan

# (c) All 4 three-way combos
strategies = ['V5_CSP', 'ETF_v3', 'ETF_v4_Wide', 'Multi_Asset']
best3_rets = {}
best3_metrics = {}
for exclude in strategies:
    include = [s for s in strategies if s != exclude]
    combo_ret = rets[include].mean(axis=1)
    m = compute_metrics(combo_ret, name=f'Best3_ex_{exclude}')
    rm = regime_split_metrics(combo_ret, regimes)
    m.update(rm)
    best3_metrics[f'ex_{exclude}'] = m
    best3_rets[f'ex_{exclude}'] = combo_ret

best_3way_key = max(best3_metrics, key=lambda k: best3_metrics[k]['sharpe'])
best_3way_ret = best3_rets[best_3way_key]
best_3way_strategies = [s for s in strategies if s not in best_3way_key]
print(f"\nBest 3-way: {best_3way_key} -> Sharpe={best3_metrics[best_3way_key]['sharpe']:.3f}")

# ============================================================
# COMBO METRICS
# ============================================================
print("\n=== COMBINED PORTFOLIOS ===")
combo_metrics = {}
combo_rets_map = {}
for label, ret in [('EqualWeight_4way', ew4_ret),
                   ('RiskParity_4way', rp4_ret),
                   (f'Best3_{best_3way_key}', best_3way_ret)]:
    ret_clean = ret.dropna()
    m = compute_metrics(ret_clean, name=label)
    rm = regime_split_metrics(ret_clean, regimes)
    m.update(rm)
    combo_metrics[label] = m
    combo_rets_map[label] = ret_clean
    print(f"  {label:35s}: Sharpe={m['sharpe']:.3f} Sortino={m['sortino']:.3f} "
          f"MaxDD={m['maxdd']:.1%} CAGR={m['cagr']:.1%} "
          f"RegimeGap={rm['regime_gap']:.3f if rm['regime_gap'] is not None else 'N/A'} "
          f"{'PASS' if rm['regime_pass'] else 'FAIL'}")

# ============================================================
# CORRELATION MATRIX
# ============================================================
corr_matrix = rets.corr()
print("\nCorrelation matrix:")
print(corr_matrix.round(3).to_string())

# ============================================================
# PERMUTATION TEST (HC #659)
# Test: does the weighting scheme beat random convex combination?
# Null: draw random Dirichlet weights over 4 strategies, compute Sharpe.
# One-sided p-value: fraction of random combos >= actual Sharpe.
# ============================================================
N_PERMS = 2000
perm_results = {}
rets_matrix = rets.values  # shape (n_days, 4)

print("\n=== PERMUTATION TESTS (HC #659) ===")
for label, ret_clean in combo_rets_map.items():
    actual_m = combo_metrics[label]
    actual_sharpe = actual_m['sharpe']

    # Random Dirichlet weights over 4 strategies
    perm_sharpe_list = []
    for _ in range(N_PERMS):
        # Random convex weights (Dirichlet alpha=[1,1,1,1] = uniform on simplex)
        w = np.random.dirichlet([1.0, 1.0, 1.0, 1.0])
        # Apply to full return matrix (use all days, same as actual period)
        # For risk parity, use same aligned dates
        if 'RiskParity' in label:
            aligned_dates = ret_clean.index
        else:
            aligned_dates = all_dates
        r_sub = rets.reindex(aligned_dates)
        perm_ret = (r_sub.values * w).sum(axis=1)
        sh_p = compute_sharpe_from_ret(perm_ret)
        perm_sharpe_list.append(sh_p)

    perm_arr = np.array(perm_sharpe_list)
    p_val = float((perm_arr >= actual_sharpe).mean())
    perm_p95 = float(np.percentile(perm_arr, 95))
    perm_p50 = float(np.percentile(perm_arr, 50))

    perm_results[label] = {
        'actual_sharpe': float(actual_sharpe),
        'perm_median': float(perm_p50),
        'perm_p95': float(perm_p95),
        'perm_mean': float(perm_arr.mean()),
        'perm_std': float(perm_arr.std()),
        'p_value': p_val,
        'significant_5pct': bool(p_val < 0.05),
        'n_perms': N_PERMS,
    }
    sig_str = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
    print(f"  [{label}] actual={actual_sharpe:.3f}, perm_p95={perm_p95:.3f}, "
          f"p={p_val:.3f} -> {sig_str}")

# ============================================================
# DAY CONCENTRATION CHECK (HC #344: cap 0.70)
# ============================================================
print("\n=== DAY CONCENTRATION (HC #344 cap=0.70) ===")
for label, m in {**individual_metrics, **combo_metrics}.items():
    conc = m.get('top1_conc')
    if conc is not None:
        flag = "PASS" if conc <= 0.70 else "FAIL"
        print(f"  {label:35s}: DayConc={conc:.2%} -> {flag}")

# ============================================================
# SAVE RESULTS
# ============================================================
def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, (np.floating,)):
        return float(obj) if not np.isnan(obj) else None
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, bool):
        return obj
    elif isinstance(obj, float) and np.isnan(obj):
        return None
    else:
        return obj

results = {
    'run_date': pd.Timestamp.now().isoformat(),
    'date_range': {
        'start': str(all_dates[0].date()),
        'end': str(all_dates[-1].date()),
        'n_days': int(n_days),
    },
    'correlation_matrix': to_serializable(corr_matrix.round(4).to_dict()),
    'individual_metrics': to_serializable(individual_metrics),
    'combo_metrics': to_serializable(combo_metrics),
    'best3_breakdown': to_serializable(best3_metrics),
    'permutation_test': to_serializable(perm_results),
    'best_3way_key': best_3way_key,
}

with open(OUTDIR / 'results_v2.json', 'w') as f:
    json.dump(results, f, indent=2)

# Save equity index curves
nav_df = pd.DataFrame({
    'V5_CSP': (1 + v5).cumprod(),
    'ETF_v3': (1 + v3).cumprod(),
    'ETF_v4_Wide': (1 + v4).cumprod(),
    'Multi_Asset': (1 + ma).cumprod(),
    'EqualWeight_4way': (1 + ew4_ret).cumprod(),
    'RiskParity_4way': (1 + rp4_ret).cumprod(),
    'Best3_combo': (1 + best_3way_ret).cumprod(),
}, index=all_dates)
nav_df.to_parquet(OUTDIR / 'nav_curves.parquet')

# Also save daily returns
rets_out = pd.DataFrame({
    'V5_CSP': v5, 'ETF_v3': v3, 'ETF_v4_Wide': v4, 'Multi_Asset': ma,
    'EqualWeight_4way': ew4_ret, 'RiskParity_4way': rp4_ret, 'Best3_combo': best_3way_ret,
}, index=all_dates)
rets_out.to_parquet(OUTDIR / 'daily_returns.parquet')

print(f"\nArtifacts saved to {OUTDIR}")
print("ALL DONE.")
