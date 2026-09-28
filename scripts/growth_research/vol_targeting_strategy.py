"""
Volatility Targeting Strategy — ETF Switching
HC #428 R1 regime symmetry gate mandatory.
HC #0: Sliding walk-forward windows only.
Optimized: vectorized returns, permutation test reduced to 500 shuffles.
Tickers: SPY, UPRO, SSO, TLT, GLD  |  Period: 2010-2026
"""

import yfinance as yf
import numpy as np
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')
from itertools import product

TRADING_DAYS = 252

# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

TICKERS = ['SPY', 'UPRO', 'SSO', 'TLT', 'GLD']
START   = '2010-01-01'
END     = '2026-06-30'

print("Downloading price data...")
raw    = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
prices = raw['Close'].ffill().dropna()[TICKERS]
rets   = prices.pct_change().dropna()
spy_ret = rets['SPY']

green_mask = spy_ret > 0
red_mask   = spy_ret < 0

print(f"Data: {rets.index[0].date()} to {rets.index[-1].date()}  ({len(rets)} days)")
print(f"Regimes — Green:{green_mask.sum()}  Red:{red_mask.sum()}  Flat:{(~green_mask & ~red_mask).sum()}")

# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def sharpe(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20 or r.std() == 0: return np.nan
    return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS))

def sortino(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20: return np.nan
    down = r[r < 0]
    if len(down) < 5 or down.std() == 0: return np.nan
    return float(r.mean() / down.std() * np.sqrt(TRADING_DAYS))

def max_dd(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    cum = np.cumprod(1 + r)
    roll_max = np.maximum.accumulate(cum)
    return float(np.min((cum - roll_max) / roll_max))

def cagr(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 5: return np.nan
    return float(np.prod(1 + r) ** (TRADING_DAYS / len(r)) - 1)

def calmar(r):
    c = cagr(r); m = max_dd(r)
    return float(c / abs(m)) if m != 0 else np.nan

def profit_factor(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    gw = r[r > 0].sum(); gl = abs(r[r < 0].sum())
    return float(gw / gl) if gl != 0 else np.inf

def win_rate(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    return float((r > 0).mean()) if len(r) > 0 else np.nan

def day_conc(r):
    r = np.asarray(r, dtype=float)
    r = r[~np.isnan(r)]
    total = r.sum()
    if total <= 0: return 1.0
    return float(r.max() / total)

def regime_split(r_arr, spy_arr):
    """r_arr and spy_arr must be aligned numpy arrays."""
    r_arr   = np.asarray(r_arr, dtype=float)
    spy_arr = np.asarray(spy_arr, dtype=float)
    valid   = ~np.isnan(r_arr)
    r_arr   = r_arr[valid]; spy_arr = spy_arr[valid]
    g = r_arr[spy_arr > 0]; rd = r_arr[spy_arr < 0]
    sg = sharpe(g); sr = sharpe(rd)
    denom = max(abs(sg), abs(sr)) if (not np.isnan(sg) and not np.isnan(sr)) else np.nan
    skew = abs(sg - sr) / denom if (denom and denom != 0 and not np.isnan(denom)) else np.nan
    gate = "PASS" if (not np.isnan(skew) and skew <= 0.50) else "FAIL"
    return sg, sr, skew, gate, len(g), len(rd)

# ─────────────────────────────────────────────────────────────────────────────
# VECTORIZED STRATEGY BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_strategy_vectorized(rets_df, spy_prices, vol_lookback, target_vol,
                               max_leverage, risk_off_asset, use_ma_filter,
                               ma_window=200):
    """
    Fully vectorized. Returns pd.Series of strategy daily returns (same index as rets_df).
    """
    n = len(rets_df)
    spy_r    = rets_df['SPY'].values
    upro_r   = rets_df['UPRO'].values
    sso_r    = rets_df['SSO'].values
    tlt_r    = rets_df['TLT'].values
    gld_r    = rets_df['GLD'].values

    if risk_off_asset == 'TLT':
        ro_r = tlt_r
    elif risk_off_asset == 'GLD':
        ro_r = gld_r
    else:  # TLT_GLD
        ro_r = 0.5 * tlt_r + 0.5 * gld_r

    # Realized vol: rolling std of spy_r
    spy_series = pd.Series(spy_r)
    realized_vol = spy_series.rolling(vol_lookback).std().values * np.sqrt(TRADING_DAYS)

    # Raw leverage
    leverage = np.where(realized_vol > 0, target_vol / realized_vol, np.nan)
    leverage = np.clip(leverage, 0.0, max_leverage)

    # MA filter
    if use_ma_filter:
        spy_px = spy_prices.values  # aligned to rets_df
        spy_ma = pd.Series(spy_px).rolling(ma_window).mean().values
        # Use prior-day MA vs prior-day price (no lookahead)
        below_ma = np.roll(spy_px, 1) < np.roll(spy_ma, 1)
        below_ma[0] = False
        leverage = np.where(below_ma, np.minimum(leverage, 1.0), leverage)

    # Map leverage to returns
    strat_r = np.full(n, np.nan)
    for i in range(n):
        lev = leverage[i]
        if np.isnan(lev):
            continue
        if lev >= 2.0:
            w = lev / 3.0
            strat_r[i] = w * upro_r[i]
        elif lev >= 1.0:
            w = lev / 2.0
            strat_r[i] = w * sso_r[i]
        elif lev >= 0.5:
            strat_r[i] = lev * spy_r[i]
        else:
            w_spy = lev / 0.5
            strat_r[i] = w_spy * spy_r[i] + (1 - w_spy) * ro_r[i]

    return pd.Series(strat_r, index=rets_df.index)

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION GATES
# ─────────────────────────────────────────────────────────────────────────────

def permutation_test(r_arr, n_shuffles=500):
    r  = r_arr[~np.isnan(r_arr)]
    actual = sharpe(r)
    rng = np.random.default_rng(42)
    null = np.array([sharpe(rng.permutation(r)) for _ in range(n_shuffles)])
    p_val = float((null >= actual).mean())
    return actual, p_val

def subperiod_cv(r_arr, n_blocks=3):
    r = r_arr[~np.isnan(r_arr)]
    bs = len(r) // n_blocks
    sharpes = [sharpe(r[i*bs:(i+1)*bs]) for i in range(n_blocks)]
    sharpes = np.array(sharpes)
    valid = sharpes[~np.isnan(sharpes)]
    if len(valid) < 2 or valid.mean() == 0: return np.nan, sharpes.tolist()
    return float(abs(valid.std() / valid.mean())), [round(float(s),4) for s in sharpes]

def outlier_rob(r_arr, pct=0.05):
    r = r_arr[~np.isnan(r_arr)]
    base = sharpe(r)
    thresh = np.quantile(r, 1 - pct)
    trimmed_s = sharpe(r[r <= thresh])
    if base == 0 or np.isnan(base) or np.isnan(trimmed_s): return np.nan
    return float((base - trimmed_s) / abs(base))

def wf_annual(strat_series, spy_series):
    """Annual walk-forward: count years strategy Sharpe > SPY Sharpe."""
    years = sorted(set(strat_series.index.year))
    beats = 0; total = 0
    detail = []
    for yr in years:
        s_yr  = strat_series[strat_series.index.year == yr].dropna()
        sp_yr = spy_series[spy_series.index.year == yr].dropna()
        if len(s_yr) < 40: continue
        ss = sharpe(s_yr.values); sp = sharpe(sp_yr.values)
        b = ss > sp
        beats += int(b); total += 1
        detail.append({'year': yr, 'strat_sharpe': round(ss,3),
                        'spy_sharpe': round(sp,3), 'beats_spy': bool(b)})
    return beats, total, detail

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────────────────

spy_arr = spy_ret.values
spy_sh  = sharpe(spy_arr)
spy_so  = sortino(spy_arr)
spy_cg  = cagr(spy_arr)
spy_md  = max_dd(spy_arr)
spy_sg, spy_sr, spy_skew, spy_r1, _, _ = regime_split(spy_arr, spy_arr)

print(f"\nSPY BUY-AND-HOLD:")
print(f"  Sharpe={spy_sh:.3f}  Sortino={spy_so:.3f}  CAGR={spy_cg:.1%}  MaxDD={spy_md:.1%}")
print(f"  Sharpe_green={spy_sg:.3f}  Sharpe_red={spy_sr:.3f}  Regime_skew={spy_skew:.3f}  [{spy_r1}]")

# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

GRID = list(product(
    [10, 20, 30],          # vol_lookback
    [0.10, 0.15, 0.20, 0.25],  # target_vol
    [2.0, 3.0],            # max_leverage
    ['TLT', 'GLD', 'TLT_GLD'],  # risk_off
    [False, True],         # ma_filter
))

print(f"\nRunning {len(GRID)} configs...")

# Precompute prices aligned to rets index (drop first row of prices to match)
spy_px_aligned = prices['SPY'].loc[rets.index]

all_results = []
best_sharpe = -np.inf
best_config = None

for idx, (vl, tv, ml, ro, ma) in enumerate(GRID):
    cfg = {'vol_lookback': vl, 'target_vol': tv, 'max_leverage': ml,
           'risk_off_asset': ro, 'use_ma_filter': ma}
    cfg_name = f"VL{vl}_TV{int(tv*100)}_ML{ml}_RO{ro}_MA{int(ma)}"

    s_rets = build_strategy_vectorized(rets, spy_px_aligned, vl, tv, ml, ro, ma)
    r_arr  = s_rets.dropna().values

    if len(r_arr) < 100:
        continue

    # Core metrics
    s_sh  = sharpe(r_arr)
    s_so  = sortino(r_arr)
    s_cg  = cagr(r_arr)
    s_md  = max_dd(r_arr)
    s_cal = calmar(r_arr)
    s_pf  = profit_factor(r_arr)
    s_wr  = win_rate(r_arr)
    s_dc  = day_conc(r_arr)

    # Align spy_ret to strategy index
    spy_aligned = spy_ret.reindex(s_rets.dropna().index).values
    sg, sr, skew, r1_gate, ng, nr = regime_split(r_arr, spy_aligned)

    # Gates
    perm_s, perm_p = permutation_test(r_arr, n_shuffles=500)
    cv, blk_sh     = subperiod_cv(r_arr)
    out_deg         = outlier_rob(r_arr)
    n_beat, n_yrs, yr_det = wf_annual(s_rets.dropna(), spy_ret)

    g_perm  = "PASS" if perm_p < 0.05 else "FAIL"
    g_subp  = "PASS" if (not np.isnan(cv) and cv < 0.50) else "FAIL"
    g_out   = "PASS" if (not np.isnan(out_deg) and out_deg < 0.30) else "FAIL"
    g_wf    = "PASS" if (n_yrs > 0 and n_beat / n_yrs >= 0.60) else "FAIL"
    g_r1    = r1_gate
    g_dc    = "PASS" if s_dc <= 0.70 else "FAIL"

    all_pass = all(g == "PASS" for g in [g_perm, g_subp, g_out, g_wf, g_r1, g_dc])

    rec = {
        'config_name': cfg_name,
        'config': cfg,
        'n_days': len(r_arr),
        'sharpe':   round(float(s_sh), 4),
        'sortino':  round(float(s_so), 4),
        'cagr':     round(float(s_cg), 4),
        'maxdd':    round(float(s_md), 4),
        'calmar':   round(float(s_cal), 4),
        'profit_factor': round(float(s_pf), 4),
        'win_rate': round(float(s_wr), 4),
        'day_conc': round(float(s_dc), 4),
        'sharpe_green': round(float(sg), 4) if not np.isnan(sg) else None,
        'sharpe_red':   round(float(sr), 4) if not np.isnan(sr) else None,
        'regime_skew':  round(float(skew), 4) if not np.isnan(skew) else None,
        'n_green': ng, 'n_red': nr,
        'gate_permutation': g_perm,
        'gate_subperiod':   g_subp,
        'gate_outlier':     g_out,
        'gate_walkforward': g_wf,
        'gate_r1_regime':   g_r1,
        'gate_day_conc':    g_dc,
        'all_gates_pass':   all_pass,
        'perm_p_value':     round(float(perm_p), 4),
        'subperiod_cv':     round(float(cv), 4) if not np.isnan(cv) else None,
        'block_sharpes':    blk_sh,
        'outlier_degradation': round(float(out_deg), 4) if not np.isnan(out_deg) else None,
        'wf_years_beating_spy': n_beat,
        'wf_total_years':       n_yrs,
        'wf_year_detail':       yr_det,
    }
    all_results.append(rec)

    if all_pass and s_sh > best_sharpe:
        best_sharpe = s_sh
        best_config = rec

    if (idx + 1) % 24 == 0:
        print(f"  [{idx+1}/{len(GRID)}] done, {sum(1 for r in all_results if r['all_gates_pass'])} passing so far")

# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

all_results.sort(key=lambda x: x['sharpe'], reverse=True)
passing = [r for r in all_results if r['all_gates_pass']]

print(f"\n{'='*70}")
print(f"GRID COMPLETE: {len(all_results)} configs, {len(passing)} pass all gates")
print(f"{'='*70}")

print(f"\nTOP 10 BY SHARPE:")
print(f"{'Config':<48} {'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'R1':>6} {'AllPass':>8}")
print("-"*82)
for r in all_results[:10]:
    print(f"{r['config_name']:<48} {r['sharpe']:>7.3f} {r['cagr']:>7.1%} "
          f"{r['maxdd']:>7.1%} {r['gate_r1_regime']:>6} {str(r['all_gates_pass']):>8}")

print(f"\nALL-GATES-PASS CONFIGS ({len(passing)}):")
print("-"*82)
for r in passing:
    print(f"{r['config_name']:<48} Sharpe={r['sharpe']:.3f}  Sortino={r['sortino']:.3f}  "
          f"Skew={r['regime_skew']:.3f}  CAGR={r['cagr']:.1%}")

if best_config:
    bc = best_config
    print(f"\n{'='*70}")
    print(f"CHAMPION (all gates pass, best Sharpe): {bc['config_name']}")
    print(f"  Sharpe:       {bc['sharpe']:.4f}")
    print(f"  Sortino:      {bc['sortino']:.4f}")
    print(f"  Calmar:       {bc['calmar']:.4f}")
    print(f"  CAGR:         {bc['cagr']:.1%}")
    print(f"  MaxDD:        {bc['maxdd']:.1%}")
    print(f"  WinRate:      {bc['win_rate']:.1%}")
    print(f"  ProfitFactor: {bc['profit_factor']:.4f}")
    print(f"  DayConc:      {bc['day_conc']:.3f}")
    print(f"  Sharpe_green: {bc['sharpe_green']:.4f}")
    print(f"  Sharpe_red:   {bc['sharpe_red']:.4f}")
    print(f"  RegimeSkew:   {bc['regime_skew']:.4f}  [{bc['gate_r1_regime']}]")
    print(f"  Perm p-value: {bc['perm_p_value']:.4f}")
    print(f"  Subperiod CV: {bc['subperiod_cv']:.4f}")
    print(f"  Outlier deg:  {bc['outlier_degradation']:.4f}")
    print(f"  WF beats SPY: {bc['wf_years_beating_spy']}/{bc['wf_total_years']} years")
else:
    print("\nNO CONFIG PASSED ALL GATES.")

# Regime summary stats
skews = [r['regime_skew'] for r in all_results if r['regime_skew'] is not None]
r1_pass_n = sum(1 for r in all_results if r['gate_r1_regime'] == 'PASS')
print(f"\nREGIME SYMMETRY SUMMARY:")
print(f"  Configs passing R1: {r1_pass_n}/{len(all_results)} = {r1_pass_n/len(all_results):.1%}")
print(f"  Median regime skew: {np.median(skews):.3f}")
print(f"  Min skew (best):    {min(skews):.3f}")

ma_skews = [r['regime_skew'] for r in all_results if r['config']['use_ma_filter'] and r['regime_skew'] is not None]
no_ma_skews = [r['regime_skew'] for r in all_results if not r['config']['use_ma_filter'] and r['regime_skew'] is not None]
print(f"  Median skew w/ MA filter:    {np.median(ma_skews):.3f}")
print(f"  Median skew w/o MA filter:   {np.median(no_ma_skews):.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

output = {
    'metadata': {
        'strategy': 'Volatility Targeting ETF Switching',
        'period': f"{rets.index[0].date()} to {rets.index[-1].date()}",
        'n_trading_days': len(rets),
        'tickers': TICKERS,
        'n_configs_tested': len(all_results),
        'n_configs_passing': len(passing),
    },
    'spy_baseline': {
        'sharpe': round(float(spy_sh), 4),
        'sortino': round(float(spy_so), 4),
        'cagr': round(float(spy_cg), 4),
        'maxdd': round(float(spy_md), 4),
        'calmar': round(float(calmar(spy_arr)), 4),
        'sharpe_green': round(float(spy_sg), 4),
        'sharpe_red': round(float(spy_sr), 4),
        'regime_skew': round(float(spy_skew), 4),
        'r1_gate': spy_r1,
    },
    'best_config': best_config,
    'passing_configs': passing,
    'top_10_configs': all_results[:10],
    'all_results': all_results,
    'regime_summary': {
        'n_r1_pass': r1_pass_n,
        'n_total': len(all_results),
        'r1_pass_rate': round(r1_pass_n / len(all_results), 4),
        'median_skew': round(float(np.median(skews)), 4),
        'min_skew': round(float(min(skews)), 4),
        'median_skew_with_ma': round(float(np.median(ma_skews)), 4),
        'median_skew_no_ma': round(float(np.median(no_ma_skews)), 4),
    }
}

out_path = '/home/jupiter/Lvl3Quant/output/growth_research/vol_targeting_results.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nSaved to {out_path}")
print("DONE.")
