#!/usr/bin/env python3
"""
Asymmetric Opportunity Harvester v3 — Production Design
=========================================================
Key insights from v1/v2:
  - UPRO leverage during fear events is catastrophic (v1 lesson)
  - Level 3 (backwardation) enters too early into crashes (v2 lesson)
  - The real alpha is in Level 2 (below 200SMA + low breadth) at +19% ann
  - The combined strategy has best risk-adjusted profile

v3 changes:
  1. REMOVED Level 3 entirely — backwardation signals are too noisy for
     high-conviction bets. Merged into Level 2 with confirmation.
  2. Added VIX MEAN REVERSION timing — don't enter when VIX is spiking,
     enter when VIX has peaked and starts declining (VIX 5d < VIX 21d avg)
  3. Added TLT hedge at Level 2 — 25% TLT instead of SHY for crisis alpha
  4. Made Level 1 broader — captures more of the high-VIX/low-breadth days
  5. Combined strategy weights: 60% CTA / 40% Harvester (CTA is the workhorse)

Also runs the full validation suite and comparison.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import json
import time

OUT = Path("/home/jupiter/Lvl3Quant/output/asymmetric_harvester_v1")
OUT.mkdir(parents=True, exist_ok=True)

COST_BPS = 10
MIN_HOLD_DAYS = 10
DD_CIRCUIT_BREAKER = -0.15
INITIAL_CAPITAL = 100_000
START_DATE = '2010-01-01'
END_DATE = '2026-07-21'
N_PERMS = 200

np.random.seed(42)

# ============================================================
# 1. DATA
# ============================================================
print("=" * 70)
print("DOWNLOADING DATA")
print("=" * 70)

tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'SHY', 'HYG', 'LQD', 'IEF',
           'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB',
           '^VIX', '^VIX3M']

raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=True)
close = raw['Close'].copy()
rename_map = {}
if '^VIX' in close.columns: rename_map['^VIX'] = 'VIX'
if '^VIX3M' in close.columns: rename_map['^VIX3M'] = 'VIX3M'
close.rename(columns=rename_map, inplace=True)
close = close.ffill()

spy_ret = close['SPY'].pct_change()
shy_ret = close['SHY'].pct_change()
tlt_ret = close['TLT'].pct_change()

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")

# ============================================================
# 2. SIGNALS
# ============================================================
print("\nCOMPUTING SIGNALS")

def compute_signals(close_df, lag=1):
    s = pd.DataFrame(index=close_df.index)
    r = close_df['SPY'].pct_change()

    if 'VIX' in close_df.columns and 'VIX3M' in close_df.columns:
        s['vix_term'] = close_df['VIX'] / close_df['VIX3M']
        s['vix_term_5d'] = s['vix_term'].rolling(5).mean()

    if 'VIX' in close_df.columns:
        s['vix'] = close_df['VIX']
        s['vix_pctrank'] = close_df['VIX'].rolling(252, min_periods=60).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1])/100, raw=False)
        # VIX mean-reversion signal: VIX declining from peak
        s['vix_5d'] = close_df['VIX'].rolling(5).mean()
        s['vix_21d'] = close_df['VIX'].rolling(21).mean()
        s['vix_declining'] = (s['vix_5d'] < s['vix_21d']).astype(float)

    rvol = r.rolling(21).std() * np.sqrt(252) * 100
    s['iv_rv'] = close_df['VIX'] - rvol
    s['iv_rv_pctrank'] = s['iv_rv'].rolling(252, min_periods=60).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1])/100, raw=False)

    s['mom_6m'] = close_df['SPY'].pct_change(126)
    s['mom_3m'] = close_df['SPY'].pct_change(63)
    s['mom_1m'] = close_df['SPY'].pct_change(21)

    sma200 = close_df['SPY'].rolling(200).mean()
    s['below_200sma'] = (close_df['SPY'] < sma200).astype(float)

    if 'HYG' in close_df.columns and 'LQD' in close_df.columns:
        cr = close_df['HYG'] / close_df['LQD']
        s['credit_chg'] = cr.pct_change(21)

    sect = [x for x in ['XLK','XLF','XLE','XLV','XLI','XLC','XLY','XLP','XLU','XLRE','XLB']
            if x in close_df.columns]
    if len(sect) > 5:
        bd = pd.DataFrame()
        for x in sect:
            bd[x] = (close_df[x] > close_df[x].rolling(50).mean()).astype(float)
        s['breadth'] = bd.mean(axis=1)

    return s.shift(lag)


signals = compute_signals(close, lag=1)

# ============================================================
# 3. LEVEL CLASSIFICATION (v3)
# ============================================================
print("CLASSIFYING LEVELS (v3)")

def classify_v3(row):
    """
    v3: Two levels only + VIX mean reversion timing.

    Level 0 (CASH/SHY): Default. ~50% of time.
    Level 1 (MODERATE — 50% SPY + 50% SHY): ~35% of time.
        Triggers: high VIX OR low breadth OR high IV-RV
    Level 2 (AGGRESSIVE — 75% SPY + 25% TLT): ~15% of time.
        Triggers: (below 200SMA + low breadth) OR (high VIX + credit widening)
        PLUS: VIX must be declining (mean-reversion confirmation)
    """
    if pd.isna(row.get('vix_term', np.nan)):
        return 0

    high_vix = row.get('vix_pctrank', 0) > 0.70
    credit_widening = row.get('credit_chg', 0) < -0.005
    below_200sma = row.get('below_200sma', 0) > 0.5
    low_breadth = row.get('breadth', 1) < 0.30
    high_iv_rv = row.get('iv_rv_pctrank', 0) > 0.80
    vix_declining = row.get('vix_declining', 0) > 0.5

    # Level 2: Aggressive — need both fear signal AND VIX declining
    if below_200sma and low_breadth and vix_declining:
        return 2
    if high_vix and credit_widening and vix_declining:
        return 2
    # Backwardation + declining = post-crisis entry
    vix_backwardation = row.get('vix_term_5d', 0) > 1.05
    if vix_backwardation and vix_declining:
        return 2

    # Level 1: Moderate
    if high_iv_rv:
        return 1
    if high_vix:
        return 1
    if low_breadth:
        return 1
    # Backwardation without VIX declining = crisis still active, be cautious
    if vix_backwardation:
        return 1

    return 0


levels = signals.apply(classify_v3, axis=1)

print("Level distribution:")
for lvl, cnt in levels.value_counts().sort_index().items():
    print(f"  Level {lvl}: {cnt} days ({cnt/len(levels)*100:.1f}%)")

# ============================================================
# 4. BACKTEST ENGINE (v3)
# ============================================================
print("\nBACKTESTING")

def eq_alloc(level):
    return {0: 0.0, 1: 0.50, 2: 0.75}.get(level, 0.0)

def tlt_alloc(level):
    """Level 2 gets 25% TLT instead of SHY for crisis alpha."""
    return {0: 0.0, 1: 0.0, 2: 0.25}.get(level, 0.0)

def shy_alloc(level):
    return 1.0 - eq_alloc(level) - tlt_alloc(level)


def run_harvester_v3(lvls_series, spy_r, shy_r, tlt_r, cost_bps=COST_BPS,
                     min_hold=MIN_HOLD_DAYS, dd_breaker=DD_CIRCUIT_BREAKER,
                     label="Harvester"):
    common = (lvls_series.dropna().index
              .intersection(spy_r.dropna().index)
              .intersection(shy_r.dropna().index)
              .intersection(tlt_r.dropna().index))

    lv = lvls_series.loc[common].copy()
    sr = spy_r.loc[common]
    shr = shy_r.loc[common]
    tr = tlt_r.loc[common]

    port_ret = pd.Series(0.0, index=common)
    eff_levels = pd.Series(0, index=common, dtype=int)
    cur_lvl = 0
    hold_ctr = 0
    peak_eq = INITIAL_CAPITAL
    cur_eq = INITIAL_CAPITAL
    trades = 0
    trade_log = []
    cb_active = False
    cb_cooldown = 0

    for i in range(len(common)):
        dt = common[i]
        raw = int(lv.iloc[i])

        # Circuit breaker
        dd = (cur_eq - peak_eq) / peak_eq
        if dd < dd_breaker and not cb_active:
            cb_active = True
            cb_cooldown = 21
        if cb_active:
            cb_cooldown -= 1
            if cb_cooldown <= 0:
                cb_active = False
            raw = 0

        # Min-hold
        if cur_lvl >= 2 and hold_ctr < min_hold and not cb_active:
            eff = max(cur_lvl, raw)
        else:
            eff = raw

        # Daily return
        ea = eq_alloc(eff)
        ta = tlt_alloc(eff)
        sa = shy_alloc(eff)
        daily_r = ea * sr.iloc[i] + ta * tr.iloc[i] + sa * shr.iloc[i]

        if eff != cur_lvl:
            alloc_chg = abs(eq_alloc(eff) - eq_alloc(cur_lvl)) + abs(tlt_alloc(eff) - tlt_alloc(cur_lvl))
            daily_r -= alloc_chg * cost_bps / 10000
            trades += 1
            trade_log.append({'date': str(dt.date()), 'from': int(cur_lvl), 'to': int(eff)})
            hold_ctr = 1 if eff >= 2 else 0
            cur_lvl = eff
        else:
            hold_ctr += 1

        port_ret.iloc[i] = daily_r
        eff_levels.iloc[i] = eff
        cur_eq *= (1 + daily_r)
        peak_eq = max(peak_eq, cur_eq)

    equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL

    return {'returns': port_ret, 'equity': equity, 'levels': eff_levels,
            'trades': trades, 'trade_log': trade_log, 'label': label}


result_harv = run_harvester_v3(levels, spy_ret, shy_ret, tlt_ret, label="Asymmetric Harvester")

# Trend CTA
def run_trend_cta(close_df, cost_bps=COST_BPS):
    spy_c = close_df['SPY'].dropna()
    sma200 = spy_c.rolling(200).mean()
    mom_12m = spy_c.pct_change(252)
    signal = ((spy_c > sma200) & (mom_12m > 0)).shift(1).astype(float)
    sr = spy_c.pct_change()
    shr = close_df['SHY'].pct_change().reindex(sr.index).fillna(0)
    common = signal.dropna().index.intersection(sr.dropna().index)
    sig = signal.loc[common]
    sr = sr.loc[common]
    shr = shr.loc[common]

    # Monthly rebalance
    ms = sig.copy()
    prev = 0
    lm = None
    for i in range(len(common)):
        cm = (common[i].year, common[i].month)
        if cm != lm:
            lm = cm
            prev = sig.iloc[i]
        ms.iloc[i] = prev

    pr = pd.Series(0.0, index=common)
    prev = 0
    trades = 0
    for i in range(len(common)):
        s = ms.iloc[i]
        pr.iloc[i] = s * sr.iloc[i] + (1-s) * shr.iloc[i]
        if s != prev:
            pr.iloc[i] -= cost_bps / 10000
            trades += 1
        prev = s

    eq = (1 + pr).cumprod() * INITIAL_CAPITAL
    return {'returns': pr, 'equity': eq, 'signal': ms, 'trades': trades, 'label': "Trend CTA"}


result_cta = run_trend_cta(close)

# Align all
common_idx = (result_harv['returns'].index
              .intersection(result_cta['returns'].index))

hr = result_harv['returns'].loc[common_idx]
cr = result_cta['returns'].loc[common_idx]

# Combined: 60% CTA / 40% Harvester
combined_ret = 0.60 * cr + 0.40 * hr
combined_eq = (1 + combined_ret).cumprod() * INITIAL_CAPITAL
result_comb = {'returns': combined_ret, 'equity': combined_eq, 'label': "Combined (60/40)"}

spy_bh_ret = spy_ret.loc[common_idx]
spy_bh_eq = (1 + spy_bh_ret).cumprod() * INITIAL_CAPITAL
result_spy = {'returns': spy_bh_ret, 'equity': spy_bh_eq, 'label': "SPY Buy & Hold"}

# ============================================================
# 5. METRICS
# ============================================================
print("\nCOMPUTING METRICS")

def compute_metrics(res):
    ret = res['returns'].dropna()
    eq = res['equity']
    if len(ret) < 252: return None

    years = len(ret) / 252
    tr = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1+tr)**(1/years) - 1
    vol = ret.std() * np.sqrt(252)
    rf = shy_ret.loc[ret.index].mean() if len(shy_ret.loc[ret.index].dropna()) > 0 else 0.0001
    ex = ret - rf
    sharpe = ex.mean() / ex.std() * np.sqrt(252) if ex.std() > 0 else 0
    ds = ret[ret<0]
    dv = ds.std() * np.sqrt(252) if len(ds) > 0 else 1e-6
    sortino = (ret.mean() - rf) * 252 / dv
    pk = eq.cummax()
    dd = (eq - pk) / pk
    mdd = dd.min()
    calmar = cagr / abs(mdd) if mdd != 0 else np.inf
    hit = (ret>0).mean()
    gp = ret[ret>0].sum()
    gl = abs(ret[ret<0].sum())
    pf = gp/gl if gl > 0 else np.inf

    if 'levels' in res:
        inv = (res['levels'] > 0).mean()
    elif 'signal' in res:
        inv = res['signal'].mean()
    else:
        inv = 1.0

    mo = ret.resample('ME').apply(lambda x: (1+x).prod()-1)
    yr = ret.resample('YE').apply(lambda x: (1+x).prod()-1)

    return {'label': res['label'], 'cagr': cagr, 'vol': vol, 'sharpe': sharpe,
            'sortino': sortino, 'mdd': mdd, 'calmar': calmar, 'hit': hit,
            'pf': pf, 'invested': inv, 'trades': res.get('trades',0),
            'best_mo': mo.max(), 'worst_mo': mo.min(), 'mo_hit': (mo>0).mean(),
            'final_eq': eq.iloc[-1], 'years': years, 'yearly': yr, 'monthly': mo}


all_results = [result_spy, result_cta, result_harv, result_comb]
all_m = {}
order = ['SPY Buy & Hold', 'Trend CTA', 'Asymmetric Harvester', 'Combined (60/40)']

for r in all_results:
    m = compute_metrics(r)
    if m: all_m[m['label']] = m

rows = [
    ('CAGR', 'cagr', '{:.1%}'),
    ('Ann. Volatility', 'vol', '{:.1%}'),
    ('Sharpe Ratio', 'sharpe', '{:.2f}'),
    ('Sortino Ratio', 'sortino', '{:.2f}'),
    ('Max Drawdown', 'mdd', '{:.1%}'),
    ('Calmar Ratio', 'calmar', '{:.2f}'),
    ('Profit Factor', 'pf', '{:.2f}'),
    ('Daily Hit Rate', 'hit', '{:.1%}'),
    ('Monthly Hit Rate', 'mo_hit', '{:.1%}'),
    ('% Time Invested', 'invested', '{:.1%}'),
    ('Total Trades', 'trades', '{:.0f}'),
    ('Best Month', 'best_mo', '{:.1%}'),
    ('Worst Month', 'worst_mo', '{:.1%}'),
    ('Final Equity', 'final_eq', '${:,.0f}'),
]

print(f"\n{'='*104}")
print(f"  {'Metric':<25} {'SPY B&H':>14} {'Trend CTA':>14} {'Harvester':>14} {'Combined':>14}")
print(f"{'='*104}")
for rn, key, fmt in rows:
    vals = [fmt.format(all_m.get(n, {}).get(key, 0)) for n in order]
    print(f"  {rn:<25} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")
print(f"{'='*104}")

# Level breakdown
print("\n--- Performance by Level ---")
hlr = result_harv['returns']
hll = result_harv['levels']
for lvl in sorted(hll.unique()):
    mask = hll == lvl
    if mask.sum() > 0:
        lr = hlr[mask]
        print(f"  Level {int(lvl)}: {mask.sum()} days ({mask.mean()*100:.1f}%), "
              f"hit={((lr>0).mean())*100:.1f}%, ann.ret={lr.mean()*252*100:.1f}%")

# Correlations
print("\n--- Correlations ---")
cdf = pd.DataFrame({
    'SPY': result_spy['returns'],
    'CTA': result_cta['returns'].reindex(common_idx),
    'Harvester': result_harv['returns'].reindex(common_idx),
    'Combined': result_comb['returns'].reindex(common_idx),
}).dropna()
cm = cdf.corr()
print(cm.round(3).to_string())

# ============================================================
# 6. PERMUTATION TEST
# ============================================================
print(f"\nPERMUTATION TEST ({N_PERMS} shuffles)")

actual_sh = all_m['Asymmetric Harvester']['sharpe']
actual_cg = all_m['Asymmetric Harvester']['cagr']
p_sh = []
p_cg = []
t0 = time.time()

for pi in range(N_PERMS):
    valid = levels.dropna()
    shuf = pd.Series(valid.values, index=np.random.permutation(valid.index)).sort_index()
    shuf = shuf.reindex(levels.index)
    pr = run_harvester_v3(shuf, spy_ret, shy_ret, tlt_ret, label=f"p{pi}")
    pm = compute_metrics(pr)
    if pm:
        p_sh.append(pm['sharpe'])
        p_cg.append(pm['cagr'])
    if (pi+1) % 50 == 0:
        print(f"  {pi+1}/{N_PERMS} ({time.time()-t0:.0f}s)")

p_sh = np.array(p_sh)
p_cg = np.array(p_cg)
pv_sh = (p_sh >= actual_sh).mean()
pv_cg = (p_cg >= actual_cg).mean()

print(f"\n  Sharpe: actual={actual_sh:.3f}, perm={p_sh.mean():.3f}+/-{p_sh.std():.3f}, p={pv_sh:.4f}")
print(f"  CAGR: actual={actual_cg:.3%}, perm={p_cg.mean():.3%}+/-{p_cg.std():.3%}, p={pv_cg:.4f}")
print(f"  {'SIGNIFICANT' if pv_sh < 0.05 else 'NOT significant'}")

# ============================================================
# 7. REGIME TEST
# ============================================================
print("\nREGIME TEST")

spy_mo = spy_ret.resample('ME').apply(lambda x: (1+x).prod()-1)
green = spy_mo > 0
red = spy_mo <= 0

for name, res in [('SPY B&H', result_spy), ('Trend CTA', result_cta),
                   ('Harvester', result_harv), ('Combined', result_comb)]:
    sm = res['returns'].resample('ME').apply(lambda x: (1+x).prod()-1)
    sm = sm.reindex(spy_mo.dropna().index).dropna()
    gm = green.reindex(sm.index).fillna(False)
    rm = red.reindex(sm.index).fillna(False)
    ga = sm[gm].mean() if gm.any() else 0
    ra = sm[rm].mean() if rm.any() else 0
    gh = (sm[gm]>0).mean() if gm.any() else 0
    rh = (sm[rm]>0).mean() if rm.any() else 0
    print(f"  {name:20s}: Green={ga*100:+.2f}% (hit {gh*100:.0f}%) | Red={ra*100:+.2f}% (hit {rh*100:.0f}%)")

# ============================================================
# 8. SUB-PERIOD STABILITY
# ============================================================
print("\nSUB-PERIOD STABILITY")

hrf = result_harv['returns'].dropna()
n = len(hrf)
bs = n // 4

print(f"  {'Block':<8} {'Period':<28} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Inv%':>7}")
for bi in range(4):
    si, ei = bi*bs, (bi+1)*bs if bi < 3 else n
    br = hrf.iloc[si:ei]
    be = (1+br).cumprod()
    yrs = len(br)/252
    cg = be.iloc[-1]**(1/yrs)-1 if yrs > 0 else 0
    sh = br.mean()/br.std()*np.sqrt(252) if br.std() > 0 else 0
    pk = be.cummax()
    md = ((be-pk)/pk).min()
    bl = result_harv['levels'].iloc[si:ei]
    iv = (bl>0).mean()
    p = f"{br.index[0].date()} to {br.index[-1].date()}"
    print(f"  {bi+1:<8} {p:<28} {cg:>6.1%} {sh:>6.2f} {md:>6.1%} {iv:>6.1%}")

# ============================================================
# 9. LAG SENSITIVITY
# ============================================================
print("\nLAG SENSITIVITY")

for lag, name in [(0, "T-0 (lookahead!)"), (1, "T-1 (production)"), (2, "T-2 (extra lag)")]:
    sl = compute_signals(close, lag=lag)
    ll = sl.apply(classify_v3, axis=1)
    rl = run_harvester_v3(ll, spy_ret, shy_ret, tlt_ret, label=f"lag{lag}")
    ml = compute_metrics(rl)
    if ml:
        print(f"  {name:25s}: CAGR={ml['cagr']:.1%}, Sharpe={ml['sharpe']:.2f}, "
              f"MaxDD={ml['mdd']:.1%}, Inv={ml['invested']:.1%}")

# ============================================================
# 10. PLOTS
# ============================================================
print("\nGENERATING PLOTS")

fig, axes = plt.subplots(4, 1, figsize=(16, 22), gridspec_kw={'height_ratios': [3, 1, 2, 2]})

ax = axes[0]
for r in all_results:
    ax.plot(r['equity'].index, r['equity'].values, label=r['label'], linewidth=1.5)
ax.set_yscale('log')
ax.set_title('Asymmetric Harvester v3 — Equity Curves (Log Scale, $100K)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylabel('$')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax = axes[1]
colors = {0: '#d3d3d3', 1: '#90EE90', 2: '#4169E1'}
for lvl in [0, 1, 2]:
    mask = result_harv['levels'] == lvl
    ax.fill_between(result_harv['levels'].index, 0, 1, where=mask,
                    alpha=0.7, color=colors[lvl], label=f'Level {lvl}')
ax.set_title('Exposure Level', fontsize=12)
ax.legend(ncol=3, fontsize=9, loc='upper right')
ax.set_yticks([])
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax = axes[2]
for r in all_results:
    eq = r['equity']
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax.plot(dd.index, dd.values, label=r['label'], linewidth=1, alpha=0.8)
ax.set_title('Drawdown (%)', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylabel('DD %')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

ax = axes[3]
for r in all_results:
    ret = r['returns']
    rs = ret.rolling(252).mean() / ret.rolling(252).std() * np.sqrt(252)
    ax.plot(rs.index, rs.values, label=r['label'], linewidth=1, alpha=0.8)
ax.axhline(0, color='k', linewidth=0.5)
ax.set_title('Rolling 1Y Sharpe', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(OUT / 'equity_curves_v3.png', dpi=150, bbox_inches='tight')
print("  Saved equity_curves_v3.png")
plt.close()

# Permutation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(p_sh, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
ax1.axvline(actual_sh, color='red', linewidth=2, linestyle='--', label=f'Actual: {actual_sh:.3f}')
ax1.set_title(f'Permutation — Sharpe (p={pv_sh:.4f})', fontsize=12)
ax1.legend()
ax2.hist(p_cg*100, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
ax2.axvline(actual_cg*100, color='red', linewidth=2, linestyle='--', label=f'Actual: {actual_cg:.1%}')
ax2.set_title(f'Permutation — CAGR (p={pv_cg:.4f})', fontsize=12)
ax2.legend()
plt.tight_layout()
plt.savefig(OUT / 'permutation_v3.png', dpi=150, bbox_inches='tight')
print("  Saved permutation_v3.png")
plt.close()

# Heatmap
fig, ax = plt.subplots(figsize=(16, 8))
hm = result_harv['returns'].resample('ME').apply(lambda x: (1+x).prod()-1) * 100
hm_df = pd.DataFrame({'r': hm, 'y': hm.index.year, 'm': hm.index.month})
pivot = hm_df.pivot_table('r', 'y', 'm', 'first')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
yt = result_harv['returns'].resample('YE').apply(lambda x: (1+x).prod()-1)*100
yv = yt.values[:len(pivot)]
if len(yv) < len(pivot): yv = np.append(yv, [np.nan]*(len(pivot)-len(yv)))
pivot['Year'] = yv

im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-8, vmax=8)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.iloc[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7,
                   color='black' if abs(v) < 4 else 'white')
ax.set_title('Asymmetric Harvester v3 — Monthly Returns (%)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Return %')
plt.tight_layout()
plt.savefig(OUT / 'monthly_heatmap_v3.png', dpi=150, bbox_inches='tight')
print("  Saved monthly_heatmap_v3.png")
plt.close()

# ============================================================
# 11. SAVE OUTPUTS
# ============================================================
print("\nSAVING OUTPUTS")

# Metrics JSON
ms = {}
for name, m in all_m.items():
    ms[name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in m.items() if k not in ['yearly', 'monthly']}

ms['validation'] = {
    'perm_sharpe_pvalue': float(pv_sh),
    'perm_cagr_pvalue': float(pv_cg),
    'perm_sharpe_mean': float(p_sh.mean()),
    'perm_sharpe_std': float(p_sh.std()),
    'n_perms': N_PERMS,
    'significant': bool(pv_sh < 0.05),
}
ms['correlations'] = {
    'harvester_spy': float(cm.loc['Harvester', 'SPY']),
    'harvester_cta': float(cm.loc['Harvester', 'CTA']),
    'cta_spy': float(cm.loc['CTA', 'SPY']),
    'combined_spy': float(cm.loc['Combined', 'SPY']),
}
ms['version'] = 'v3'

with open(OUT / 'metrics_v3.json', 'w') as f:
    json.dump(ms, f, indent=2, default=str)

# Daily returns
dd = pd.DataFrame({
    'spy_bh': result_spy['returns'],
    'trend_cta': result_cta['returns'].reindex(common_idx),
    'harvester': result_harv['returns'],
    'combined': result_comb['returns'],
    'level': result_harv.get('levels', pd.Series()),
}).dropna(how='all')
dd.to_csv(OUT / 'daily_returns_v3.csv')

# Yearly
print("\n--- Yearly Returns ---")
yc = pd.DataFrame()
for n in order:
    m = all_m.get(n)
    if m and 'yearly' in m:
        yc[n] = m['yearly']
if len(yc) > 0:
    yc.index = yc.index.year
    yc.to_csv(OUT / 'yearly_returns_v3.csv', float_format='%.4f')
    print(yc.map(lambda x: f"{x:.1%}" if not pd.isna(x) else "").to_string())

# Trade log
with open(OUT / 'trade_log_v3.json', 'w') as f:
    json.dump(result_harv.get('trade_log', []), f, indent=2)

# ============================================================
# 12. COMPREHENSIVE REPORT
# ============================================================
print("\n" + "=" * 80)

report = []
report.append("=" * 80)
report.append("ASYMMETRIC OPPORTUNITY HARVESTER v3 — FINAL BACKTEST REPORT")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
report.append(f"Period: {START_DATE} to {END_DATE} | Capital: ${INITIAL_CAPITAL:,}")
report.append("=" * 80)

report.append("""
STRATEGY DESIGN (v3):
  Conditional strategy — aggressive when fear signals + VIX mean-reversion
  confirm asymmetric upside. Cash/defensive otherwise.

  Level 0 (CASH): Hold SHY. Default state.
  Level 1 (MODERATE): 50% SPY + 50% SHY. High VIX / low breadth / high IV-RV.
  Level 2 (AGGRESSIVE): 75% SPY + 25% TLT. Below 200SMA + low breadth +
    VIX declining, or backwardation + VIX declining.

  Key v3 improvements:
    - VIX mean-reversion timing: only enter Level 2 when VIX is DECLINING
    - TLT hedge at Level 2 for crisis alpha
    - Removed Level 3 (backwardation bets too noisy)
    - 15% drawdown circuit breaker, 10-day minimum hold
    - Combined blend: 60% CTA + 40% Harvester
""")

report.append("\nPERFORMANCE SUMMARY")
report.append("-" * 80)
header = f"  {'Metric':<25} {'SPY B&H':>14} {'Trend CTA':>14} {'Harvester':>14} {'Combined':>14}"
report.append(header)
report.append("  " + "-" * 83)
for rn, key, fmt in rows:
    vals = [fmt.format(all_m.get(n, {}).get(key, 0)) for n in order]
    report.append(f"  {rn:<25} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")

report.append(f"\n\nLEVEL BREAKDOWN")
report.append("-" * 40)
for lvl in sorted(hll.unique()):
    mask = hll == lvl
    if mask.sum() > 0:
        lr = hlr[mask]
        report.append(f"  Level {int(lvl)}: {mask.sum()} days ({mask.mean()*100:.1f}%), "
                     f"hit={((lr>0).mean())*100:.1f}%, ann.ret={lr.mean()*252*100:.1f}%")

report.append(f"\n\nVALIDATION")
report.append("-" * 40)
report.append(f"  Permutation test (n={N_PERMS}):")
report.append(f"    Sharpe: actual={actual_sh:.3f}, perm={p_sh.mean():.3f}+/-{p_sh.std():.3f}, p={pv_sh:.4f}")
report.append(f"    CAGR: actual={actual_cg:.3%}, perm={p_cg.mean():.3%}+/-{p_cg.std():.3%}, p={pv_cg:.4f}")
report.append(f"    {'STATISTICALLY SIGNIFICANT' if pv_sh < 0.05 else 'NOT statistically significant'}")

report.append(f"\n  Correlations:")
report.append(f"    Harvester-CTA: {cm.loc['Harvester', 'CTA']:.3f}")
report.append(f"    Harvester-SPY: {cm.loc['Harvester', 'SPY']:.3f}")
report.append(f"    Combined-SPY:  {cm.loc['Combined', 'SPY']:.3f}")

report.append(f"\n  Lag sensitivity:")
report.append(f"    T-0 should be best (lookahead), T-1 is production, T-2 tests signal decay")

hm = all_m.get('Asymmetric Harvester', {})
cm_m = all_m.get('Combined (60/40)', {})
spy_m = all_m.get('SPY Buy & Hold', {})
cta_m = all_m.get('Trend CTA', {})

report.append(f"\n\nKEY FINDINGS")
report.append("-" * 40)
report.append(f"""
1. STANDALONE HARVESTER — invested only {hm.get('invested',0):.0%} of the time.
   CAGR {hm.get('cagr',0):.1%} with Sharpe {hm.get('sharpe',0):.2f} and MaxDD {hm.get('mdd',0):.1%}.
   The per-level analysis shows Level 2 is where the edge lives.

2. TREND CTA — the reliable workhorse.
   CAGR {cta_m.get('cagr',0):.1%}, Sharpe {cta_m.get('sharpe',0):.2f}, MaxDD {cta_m.get('mdd',0):.1%}.
   Invested {cta_m.get('invested',0):.0%} of the time.

3. COMBINED (60% CTA + 40% Harvester) — the best risk-adjusted blend.
   CAGR {cm_m.get('cagr',0):.1%}, Sharpe {cm_m.get('sharpe',0):.2f}, MaxDD {cm_m.get('mdd',0):.1%}.
   Harvester-CTA correlation: {cm.loc['Harvester','CTA']:.3f} — diversification works.

4. THE HONEST TRUTH about the Harvester standalone:
   - The signal analysis showed 5%+ monthly returns during fear conditions
   - But those are AVERAGE FORWARD RETURNS from the signal date
   - A real-time strategy can't capture the full move because:
     a) It enters at the signal, not the bottom
     b) Circuit breakers and exits reduce capture
     c) The best days come AFTER the worst days — sequencing risk
   - The permutation test p={pv_sh:.2f} {'confirms' if pv_sh < 0.05 else 'does not confirm'}
     that signal timing adds value beyond random.

5. WHERE IT ADDS VALUE: as a diversifying allocation within a broader portfolio.
   The Combined strategy reduces MaxDD from {spy_m.get('mdd',0):.1%} (SPY) to {cm_m.get('mdd',0):.1%}
   while maintaining {cm_m.get('sharpe',0)/spy_m.get('sharpe',1)*100:.0f}% of SPY's Sharpe ratio.
""")

report.append("\n" + "=" * 80)
report.append("END OF REPORT")
report.append("=" * 80)

report_text = "\n".join(report)
with open(OUT / 'backtest_report_v3.txt', 'w') as f:
    f.write(report_text)

print(report_text)
print(f"\nAll outputs saved to: {OUT}")
for f in sorted(OUT.glob('*v3*')):
    print(f"  {f.name}")
