#!/usr/bin/env python3
"""
Asymmetric Opportunity Harvester v2 — Fixed Design
====================================================
Key fixes from v1:
1. REMOVED UPRO — 3x leverage during fear events = catastrophic.
   Level 3 now uses 100% SPY (still max equity, just not leveraged).
2. Added CONFIRMATION DELAY — don't buy the first day of backwardation,
   wait 5 days for initial crash to stabilize.
3. Reduced min-hold from 21 to 10 days — allows faster exit if conditions reverse.
4. Added DRAWDOWN CIRCUIT BREAKER — if portfolio DD > 15%, force to Level 0.

Also builds Trend CTA and 50/50 Combined.

Full validation suite: permutation test, regime, sub-period, lag sensitivity.
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
MIN_HOLD_DAYS = 10  # Reduced from 21
CONFIRM_DAYS = 5    # Wait 5 days after signal triggers before entering L2/L3
DD_CIRCUIT_BREAKER = -0.15  # Force to cash if DD > 15%
INITIAL_CAPITAL = 100_000
START_DATE = '2010-01-01'
END_DATE = '2026-07-21'

np.random.seed(42)

# ============================================================
# 1. DOWNLOAD DATA
# ============================================================
print("=" * 70)
print("STEP 1: Downloading data")
print("=" * 70)

tickers = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'SHY', 'HYG', 'LQD', 'IEF',
           'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB',
           '^VIX', '^VIX3M']

raw = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=True)
close = raw['Close'].copy()

rename_map = {}
if '^VIX' in close.columns:
    rename_map['^VIX'] = 'VIX'
if '^VIX3M' in close.columns:
    rename_map['^VIX3M'] = 'VIX3M'
close.rename(columns=rename_map, inplace=True)
close = close.ffill()

spy_ret = close['SPY'].pct_change()
shy_ret = close['SHY'].pct_change()

sector_etfs = [s for s in ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
               if s in close.columns]

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days")

# ============================================================
# 2. COMPUTE SIGNALS (T-1 shifted)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Computing signals")
print("=" * 70)

def compute_signals(close_df, lag=1):
    """Compute all signals with specified lag."""
    sigs = pd.DataFrame(index=close_df.index)
    spy_r = close_df['SPY'].pct_change()

    # VIX term structure
    if 'VIX' in close_df.columns and 'VIX3M' in close_df.columns:
        sigs['vix_term_ratio'] = close_df['VIX'] / close_df['VIX3M']

    # VIX level + rolling percentile
    if 'VIX' in close_df.columns:
        sigs['vix_level'] = close_df['VIX']
        sigs['vix_pctrank'] = close_df['VIX'].rolling(252, min_periods=60).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100, raw=False
        )

    # IV-RV spread
    spy_rvol = spy_r.rolling(21).std() * np.sqrt(252) * 100
    sigs['iv_rv_spread'] = close_df['VIX'] - spy_rvol
    sigs['iv_rv_pctrank'] = sigs['iv_rv_spread'].rolling(252, min_periods=60).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100, raw=False
    )

    # Momentum
    sigs['spy_mom_6m'] = close_df['SPY'].pct_change(126)
    sigs['spy_mom_3m'] = close_df['SPY'].pct_change(63)
    sigs['spy_mom_1m'] = close_df['SPY'].pct_change(21)

    # SMA
    sma200 = close_df['SPY'].rolling(200).mean()
    sigs['spy_below_200sma'] = (close_df['SPY'] < sma200).astype(float)

    # Credit
    if 'HYG' in close_df.columns and 'LQD' in close_df.columns:
        credit_ratio = close_df['HYG'] / close_df['LQD']
        sigs['credit_21d_chg'] = credit_ratio.pct_change(21)

    # Breadth
    sect = [s for s in ['XLK','XLF','XLE','XLV','XLI','XLC','XLY','XLP','XLU','XLRE','XLB']
            if s in close_df.columns]
    if len(sect) > 5:
        b_df = pd.DataFrame()
        for s in sect:
            b_df[s] = (close_df[s] > close_df[s].rolling(50).mean()).astype(float)
        sigs['breadth'] = b_df.mean(axis=1)

    # VIX 5-day moving average of term structure (for confirmation delay)
    if 'vix_term_ratio' in sigs.columns:
        sigs['vix_term_5d_avg'] = sigs['vix_term_ratio'].rolling(5).mean()

    # SPY 10-day realized return (for momentum confirmation)
    sigs['spy_ret_10d'] = close_df['SPY'].pct_change(10)

    return sigs.shift(lag)


signals = compute_signals(close, lag=1)
print(f"Computed {len(signals.columns)} signals")

# ============================================================
# 3. CLASSIFY LEVELS (v2 — with confirmation delay)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Classifying exposure levels (v2)")
print("=" * 70)

def classify_level_v2(row):
    """
    v2 level classification with confirmation delays.

    Key changes from v1:
    - Level 3 requires 5-day average backwardation (not just 1 day)
    - This avoids entering on the first day of a spike
    """
    if pd.isna(row.get('vix_term_ratio', np.nan)):
        return 0

    # Use 5-day avg for backwardation confirmation
    vix_backwardation_confirmed = row.get('vix_term_5d_avg', 0) > 1.05
    vix_backwardation_raw = row.get('vix_term_ratio', 0) > 1.05
    weak_momentum = row.get('spy_mom_6m', 0) < 0
    high_vix = row.get('vix_pctrank', 0) > 0.70
    credit_widening = row.get('credit_21d_chg', 0) < -0.005
    below_200sma = row.get('spy_below_200sma', 0) > 0.5
    low_breadth = row.get('breadth', 1) < 0.30
    high_iv_rv = row.get('iv_rv_pctrank', 0) > 0.80

    # Level 3: Confirmed backwardation (5d avg) + weak momentum
    if vix_backwardation_confirmed and weak_momentum:
        return 3
    if vix_backwardation_confirmed:
        return 3

    # Level 2: Aggressive
    if below_200sma and low_breadth:
        return 2
    if high_vix and credit_widening:
        return 2

    # Level 1: Moderate
    if high_iv_rv:
        return 1
    if high_vix:
        return 1
    if low_breadth:
        return 1

    return 0


levels = signals.apply(classify_level_v2, axis=1)
levels.name = 'level'

print("Exposure level distribution:")
for lvl, cnt in levels.value_counts().sort_index().items():
    print(f"  Level {lvl}: {cnt} days ({cnt/len(levels)*100:.1f}%)")

# ============================================================
# 4. BACKTEST ENGINE (v2 — no UPRO, with circuit breaker)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Backtesting")
print("=" * 70)

def _equity_alloc(level):
    """Equity allocation by level. NO leverage."""
    return {0: 0.0, 1: 0.50, 2: 0.75, 3: 1.0}.get(level, 0.0)


def run_harvester_v2(levels_series, spy_r, shy_r, cost_bps=COST_BPS,
                     min_hold=MIN_HOLD_DAYS, dd_breaker=DD_CIRCUIT_BREAKER,
                     label="Harvester"):
    """
    v2 backtest: no UPRO, with drawdown circuit breaker.
    Level 3 = 100% SPY (not 3x).
    """
    common = levels_series.dropna().index.intersection(spy_r.dropna().index).intersection(shy_r.dropna().index)
    lvls = levels_series.loc[common].copy()
    sr = spy_r.loc[common]
    sh = shy_r.loc[common]

    port_ret = pd.Series(0.0, index=common)
    eff_levels = pd.Series(0, index=common, dtype=int)
    current_level = 0
    hold_counter = 0
    peak_equity = INITIAL_CAPITAL
    current_equity = INITIAL_CAPITAL
    trades = 0
    trade_log = []
    circuit_breaker_active = False
    circuit_breaker_cooldown = 0

    for i in range(len(common)):
        dt = common[i]
        raw_level = int(lvls.iloc[i])

        # Circuit breaker check
        dd_pct = (current_equity - peak_equity) / peak_equity
        if dd_pct < dd_breaker and not circuit_breaker_active:
            circuit_breaker_active = True
            circuit_breaker_cooldown = 21  # Stay in cash for 21 days

        if circuit_breaker_active:
            circuit_breaker_cooldown -= 1
            if circuit_breaker_cooldown <= 0:
                circuit_breaker_active = False
            raw_level = 0  # Force cash

        # Min-hold constraint for Level 2-3
        if current_level >= 2 and hold_counter < min_hold and not circuit_breaker_active:
            effective_level = max(current_level, raw_level)
        else:
            effective_level = raw_level

        # Compute daily return
        eq_alloc = _equity_alloc(effective_level)
        daily_r = eq_alloc * sr.iloc[i] + (1 - eq_alloc) * sh.iloc[i]

        # Trading cost
        if effective_level != current_level:
            alloc_change = abs(_equity_alloc(effective_level) - _equity_alloc(current_level))
            cost = alloc_change * cost_bps / 10000
            daily_r -= cost
            trades += 1
            trade_log.append({
                'date': str(dt.date()),
                'from': int(current_level),
                'to': int(effective_level),
            })
            hold_counter = 1 if effective_level >= 2 else 0
            current_level = effective_level
        else:
            hold_counter += 1

        port_ret.iloc[i] = daily_r
        eff_levels.iloc[i] = effective_level

        current_equity *= (1 + daily_r)
        peak_equity = max(peak_equity, current_equity)

    equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL

    return {
        'returns': port_ret,
        'equity': equity,
        'levels': eff_levels,
        'trades': trades,
        'trade_log': trade_log,
        'label': label,
    }


# Run Harvester v2
result_harvester = run_harvester_v2(levels, spy_ret, shy_ret, label="Asymmetric Harvester")

# Trend CTA
def run_trend_cta(close_df, cost_bps=COST_BPS, label="Trend CTA"):
    spy_close = close_df['SPY'].dropna()
    sma200 = spy_close.rolling(200).mean()
    mom_12m = spy_close.pct_change(252)
    trend_signal = ((spy_close > sma200) & (mom_12m > 0)).shift(1).astype(float)

    spy_r = spy_close.pct_change()
    shy_r = close_df['SHY'].pct_change().reindex(spy_r.index).fillna(0)
    common = trend_signal.dropna().index.intersection(spy_r.dropna().index)

    sig = trend_signal.loc[common]
    sr = spy_r.loc[common]
    shr = shy_r.loc[common]

    # Monthly rebalance
    prev = 0
    last_month = None
    port_ret = pd.Series(0.0, index=common)
    monthly_sig = sig.copy()
    trades = 0

    for i in range(len(common)):
        dt = common[i]
        cm = (dt.year, dt.month)
        if cm != last_month:
            last_month = cm
            prev_sig = sig.iloc[i]
        monthly_sig.iloc[i] = prev_sig

    prev = 0
    for i in range(len(common)):
        s = monthly_sig.iloc[i]
        daily_r = s * sr.iloc[i] + (1 - s) * shr.iloc[i]
        if s != prev:
            daily_r -= cost_bps / 10000
            trades += 1
        port_ret.iloc[i] = daily_r
        prev = s

    equity = (1 + port_ret).cumprod() * INITIAL_CAPITAL
    return {'returns': port_ret, 'equity': equity, 'signal': monthly_sig,
            'trades': trades, 'label': label}


result_cta = run_trend_cta(close)

# Combined 50/50
common_idx = result_harvester['returns'].index.intersection(result_cta['returns'].index)
harv_ret = result_harvester['returns'].loc[common_idx]
cta_ret = result_cta['returns'].loc[common_idx]
combined_ret = 0.50 * harv_ret + 0.50 * cta_ret
combined_equity = (1 + combined_ret).cumprod() * INITIAL_CAPITAL
result_combined = {'returns': combined_ret, 'equity': combined_equity, 'label': "Combined (50/50)"}

# SPY B&H
spy_bh_ret = spy_ret.loc[common_idx]
spy_bh_equity = (1 + spy_bh_ret).cumprod() * INITIAL_CAPITAL
result_spy = {'returns': spy_bh_ret, 'equity': spy_bh_equity, 'label': "SPY Buy & Hold"}

# ============================================================
# 5. METRICS
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Performance metrics")
print("=" * 70)

def compute_metrics(res):
    ret = res['returns'].dropna()
    eq = res['equity']
    if len(ret) < 252:
        return None

    years = len(ret) / 252
    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / years) - 1
    ann_vol = ret.std() * np.sqrt(252)

    rf_daily = shy_ret.loc[ret.index].mean() if len(shy_ret.loc[ret.index].dropna()) > 0 else 0.0001
    excess = ret - rf_daily
    sharpe = excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    downside = ret[ret < 0]
    dv = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = (ret.mean() - rf_daily) * 252 / dv

    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.inf

    hit = (ret > 0).mean()
    gp = ret[ret > 0].sum()
    gl = abs(ret[ret < 0].sum())
    pf = gp / gl if gl > 0 else np.inf

    if 'levels' in res:
        lvls = res['levels']
        pct_inv = (lvls > 0).mean()
    elif 'signal' in res:
        pct_inv = res['signal'].mean()
    else:
        pct_inv = 1.0

    monthly = ret.resample('ME').apply(lambda x: (1 + x).prod() - 1)
    yearly = ret.resample('YE').apply(lambda x: (1 + x).prod() - 1)

    return {
        'label': res['label'],
        'total_return': total_return, 'cagr': cagr, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'sortino': sortino, 'max_dd': max_dd, 'calmar': calmar,
        'hit_rate': hit, 'profit_factor': pf, 'pct_invested': pct_inv,
        'trades': res.get('trades', 0),
        'best_month': monthly.max(), 'worst_month': monthly.min(),
        'monthly_hit': (monthly > 0).mean(),
        'final_equity': eq.iloc[-1], 'years': years,
        'yearly_returns': yearly, 'monthly_returns': monthly,
    }


all_results = [result_spy, result_cta, result_harvester, result_combined]
all_metrics = {}
metrics_order = ['SPY Buy & Hold', 'Trend CTA', 'Asymmetric Harvester', 'Combined (50/50)']

for res in all_results:
    m = compute_metrics(res)
    if m:
        all_metrics[m['label']] = m

metric_rows = [
    ('CAGR', 'cagr', '{:.1%}'),
    ('Ann. Volatility', 'ann_vol', '{:.1%}'),
    ('Sharpe Ratio', 'sharpe', '{:.2f}'),
    ('Sortino Ratio', 'sortino', '{:.2f}'),
    ('Max Drawdown', 'max_dd', '{:.1%}'),
    ('Calmar Ratio', 'calmar', '{:.2f}'),
    ('Profit Factor', 'profit_factor', '{:.2f}'),
    ('Daily Hit Rate', 'hit_rate', '{:.1%}'),
    ('Monthly Hit Rate', 'monthly_hit', '{:.1%}'),
    ('% Time Invested', 'pct_invested', '{:.1%}'),
    ('Total Trades', 'trades', '{:.0f}'),
    ('Best Month', 'best_month', '{:.1%}'),
    ('Worst Month', 'worst_month', '{:.1%}'),
    ('Final Equity', 'final_equity', '${:,.0f}'),
]

print(f"\n{'='*100}")
print(f"{'Metric':<25} {'SPY B&H':>14} {'Trend CTA':>14} {'Harvester':>14} {'Combined':>14}")
print(f"{'='*100}")
for rn, key, fmt in metric_rows:
    vals = [fmt.format(all_metrics.get(n, {}).get(key, 0)) for n in metrics_order]
    print(f"  {rn:<23} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")
print(f"{'='*100}")

# Level breakdown
print("\n--- Harvester: Performance by Level ---")
hr = result_harvester['returns']
hl = result_harvester['levels']
for lvl in sorted(hl.unique()):
    mask = hl == lvl
    if mask.sum() > 0:
        lr = hr[mask]
        print(f"  Level {int(lvl)}: {mask.sum()} days ({mask.mean()*100:.1f}%), "
              f"hit={((lr>0).mean())*100:.1f}%, ann.ret={lr.mean()*252*100:.1f}%")

# Correlations
print("\n--- Strategy Correlations ---")
corr_df = pd.DataFrame({
    'SPY': result_spy['returns'],
    'CTA': result_cta['returns'].reindex(common_idx),
    'Harvester': result_harvester['returns'].reindex(common_idx),
    'Combined': result_combined['returns'].reindex(common_idx),
}).dropna()
corr_matrix = corr_df.corr()
print(corr_matrix.round(3).to_string())

# ============================================================
# 6. PERMUTATION TEST (200 shuffles)
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Permutation test")
print("=" * 70)

N_PERMS = 200
actual_sharpe = all_metrics['Asymmetric Harvester']['sharpe']
actual_cagr = all_metrics['Asymmetric Harvester']['cagr']
perm_sharpes = []
perm_cagrs = []

print(f"Actual: Sharpe={actual_sharpe:.3f}, CAGR={actual_cagr:.1%}")
t0 = time.time()

for pi in range(N_PERMS):
    # Shuffle signal dates
    valid = levels.dropna()
    shuffled = pd.Series(valid.values, index=np.random.permutation(valid.index)).sort_index()
    shuffled = shuffled.reindex(levels.index)

    pr = run_harvester_v2(shuffled, spy_ret, shy_ret, label=f"perm_{pi}")
    pm = compute_metrics(pr)
    if pm:
        perm_sharpes.append(pm['sharpe'])
        perm_cagrs.append(pm['cagr'])

    if (pi+1) % 50 == 0:
        print(f"  {pi+1}/{N_PERMS} ({time.time()-t0:.0f}s)")

perm_sharpes = np.array(perm_sharpes)
perm_cagrs = np.array(perm_cagrs)
p_sharpe = (perm_sharpes >= actual_sharpe).mean()
p_cagr = (perm_cagrs >= actual_cagr).mean()

print(f"\nPermutation results:")
print(f"  Sharpe: actual={actual_sharpe:.3f}, perm_mean={perm_sharpes.mean():.3f}, p={p_sharpe:.4f}")
print(f"  CAGR: actual={actual_cagr:.3%}, perm_mean={perm_cagrs.mean():.3%}, p={p_cagr:.4f}")
print(f"  {'** SIGNIFICANT **' if p_sharpe < 0.05 else '!! NOT significant !!'}")

# ============================================================
# 7. REGIME TEST
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Regime test (green vs red months)")
print("=" * 70)

spy_monthly = spy_ret.resample('ME').apply(lambda x: (1+x).prod()-1)
common_m = spy_monthly.dropna().index

green = spy_monthly > 0
red = spy_monthly <= 0

for name, res in [('SPY B&H', result_spy), ('Trend CTA', result_cta),
                   ('Harvester', result_harvester), ('Combined', result_combined)]:
    sm = res['returns'].resample('ME').apply(lambda x: (1+x).prod()-1)
    sm = sm.reindex(common_m).dropna()
    gm = green.reindex(sm.index).fillna(False)
    rm = red.reindex(sm.index).fillna(False)
    g_avg = sm[gm].mean() if gm.any() else 0
    r_avg = sm[rm].mean() if rm.any() else 0
    g_hit = (sm[gm]>0).mean() if gm.any() else 0
    r_hit = (sm[rm]>0).mean() if rm.any() else 0
    print(f"  {name:20s}: Green={g_avg*100:+.2f}% (hit {g_hit*100:.0f}%) | Red={r_avg*100:+.2f}% (hit {r_hit*100:.0f}%)")

# ============================================================
# 8. SUB-PERIOD STABILITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Sub-period stability (4 blocks)")
print("=" * 70)

hrf = result_harvester['returns'].dropna()
n = len(hrf)
bs = n // 4

print(f"{'Block':<10} {'Period':<28} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>7} {'Inv%':>7}")
print("-" * 70)

for bi in range(4):
    si = bi * bs
    ei = si + bs if bi < 3 else n
    br = hrf.iloc[si:ei]
    beq = (1 + br).cumprod()
    yrs = len(br) / 252
    cagr_b = beq.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else 0
    sharpe_b = br.mean() / br.std() * np.sqrt(252) if br.std() > 0 else 0
    pk = beq.cummax()
    mdd_b = ((beq - pk) / pk).min()
    bl = result_harvester['levels'].iloc[si:ei]
    inv_b = (bl > 0).mean()
    period = f"{br.index[0].date()} to {br.index[-1].date()}"
    print(f"  {bi+1:<8} {period:<28} {cagr_b:>6.1%} {sharpe_b:>6.2f} {mdd_b:>6.1%} {inv_b:>6.1%}")

# ============================================================
# 9. LAG SENSITIVITY
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Lag sensitivity")
print("=" * 70)

for lag, name in [(0, "T-0 (lookahead!)"), (1, "T-1 (production)"), (2, "T-2 (extra lag)")]:
    sigs_l = compute_signals(close, lag=lag)
    lvls_l = sigs_l.apply(classify_level_v2, axis=1)
    rl = run_harvester_v2(lvls_l, spy_ret, shy_ret, label=f"lag_{lag}")
    ml = compute_metrics(rl)
    if ml:
        print(f"  {name:25s}: CAGR={ml['cagr']:.1%}, Sharpe={ml['sharpe']:.2f}, "
              f"MaxDD={ml['max_dd']:.1%}, Invested={ml['pct_invested']:.1%}")

# ============================================================
# 10. PLOTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 10: Generating plots")
print("=" * 70)

fig, axes = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [3, 1, 2, 2]})

# Equity curves
ax1 = axes[0]
for res in all_results:
    ax1.plot(res['equity'].index, res['equity'].values, label=res['label'], linewidth=1.5)
ax1.set_yscale('log')
ax1.set_title('Equity Curves — Asymmetric Harvester v2 (Log Scale)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('Portfolio Value ($)')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Exposure levels
ax2 = axes[1]
colors = {0: '#d3d3d3', 1: '#90EE90', 2: '#4169E1', 3: '#FF4500'}
for lvl in [0, 1, 2, 3]:
    mask = result_harvester['levels'] == lvl
    ax2.fill_between(result_harvester['levels'].index, 0, 1, where=mask,
                    alpha=0.7, color=colors[lvl], label=f'Level {lvl}')
ax2.set_title('Exposure Level', fontsize=12)
ax2.legend(ncol=4, fontsize=9, loc='upper right')
ax2.set_yticks([])
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Drawdowns
ax3 = axes[2]
for res in all_results:
    eq = res['equity']
    dd = (eq - eq.cummax()) / eq.cummax() * 100
    ax3.plot(dd.index, dd.values, label=res['label'], linewidth=1, alpha=0.8)
ax3.set_title('Drawdown (%)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylabel('DD %')
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# Rolling Sharpe
ax4 = axes[3]
for res in all_results:
    r = res['returns']
    rs = r.rolling(252).mean() / r.rolling(252).std() * np.sqrt(252)
    ax4.plot(rs.index, rs.values, label=res['label'], linewidth=1, alpha=0.8)
ax4.axhline(0, color='k', linewidth=0.5)
ax4.set_title('Rolling 1Y Sharpe', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(OUT / 'equity_curves_v2.png', dpi=150, bbox_inches='tight')
print("  Saved equity_curves_v2.png")
plt.close()

# Permutation distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(perm_sharpes, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
ax1.axvline(actual_sharpe, color='red', linewidth=2, linestyle='--', label=f'Actual: {actual_sharpe:.3f}')
ax1.set_title(f'Permutation — Sharpe (p={p_sharpe:.4f})', fontsize=12)
ax1.legend()
ax2.hist(perm_cagrs*100, bins=30, alpha=0.7, color='steelblue', edgecolor='white')
ax2.axvline(actual_cagr*100, color='red', linewidth=2, linestyle='--', label=f'Actual: {actual_cagr:.1%}')
ax2.set_title(f'Permutation — CAGR (p={p_cagr:.4f})', fontsize=12)
ax2.legend()
plt.tight_layout()
plt.savefig(OUT / 'permutation_v2.png', dpi=150, bbox_inches='tight')
print("  Saved permutation_v2.png")
plt.close()

# Monthly heatmap
fig, ax = plt.subplots(figsize=(16, 8))
hm = result_harvester['returns'].resample('ME').apply(lambda x: (1+x).prod()-1) * 100
hm_df = pd.DataFrame({'r': hm, 'year': hm.index.year, 'month': hm.index.month})
pivot = hm_df.pivot_table('r', 'year', 'month', 'first')
pivot.columns = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

# Add year total
yt = result_harvester['returns'].resample('YE').apply(lambda x: (1+x).prod()-1) * 100
yr_vals = yt.values[:len(pivot)]
if len(yr_vals) < len(pivot):
    yr_vals = np.append(yr_vals, [np.nan] * (len(pivot) - len(yr_vals)))
pivot['Year'] = yr_vals

im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, fontsize=9)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        v = pivot.iloc[i, j]
        if not np.isnan(v):
            ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7,
                   color='black' if abs(v) < 5 else 'white')
ax.set_title('Asymmetric Harvester v2 — Monthly Returns (%)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='Return %')
plt.tight_layout()
plt.savefig(OUT / 'monthly_heatmap_v2.png', dpi=150, bbox_inches='tight')
print("  Saved monthly_heatmap_v2.png")
plt.close()

# ============================================================
# 11. SAVE ALL OUTPUTS
# ============================================================
print("\n" + "=" * 70)
print("STEP 11: Saving outputs")
print("=" * 70)

# Metrics JSON
ms = {}
for name, m in all_metrics.items():
    ms[name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in m.items() if k not in ['yearly_returns', 'monthly_returns']}

ms['validation'] = {
    'permutation_sharpe_pvalue': float(p_sharpe),
    'permutation_cagr_pvalue': float(p_cagr),
    'perm_sharpe_mean': float(perm_sharpes.mean()),
    'perm_sharpe_std': float(perm_sharpes.std()),
    'n_perms': N_PERMS,
    'significant': bool(p_sharpe < 0.05),
}
ms['correlations'] = {
    'harvester_vs_spy': float(corr_matrix.loc['Harvester', 'SPY']),
    'harvester_vs_cta': float(corr_matrix.loc['Harvester', 'CTA']),
    'cta_vs_spy': float(corr_matrix.loc['CTA', 'SPY']),
}
ms['version'] = 'v2'
ms['changes_from_v1'] = [
    'Removed UPRO (3x leverage) — Level 3 now 100% SPY',
    'Added 5-day confirmation for backwardation signals',
    'Added 15% drawdown circuit breaker (21-day cash cooldown)',
    'Reduced min-hold from 21 to 10 days',
]

with open(OUT / 'metrics_v2.json', 'w') as f:
    json.dump(ms, f, indent=2, default=str)

# Daily returns CSV
dd = pd.DataFrame({
    'spy_bh': result_spy['returns'],
    'trend_cta': result_cta['returns'].reindex(common_idx),
    'harvester': result_harvester['returns'],
    'combined': result_combined['returns'],
    'level': result_harvester.get('levels', pd.Series()),
}).dropna(how='all')
dd.to_csv(OUT / 'daily_returns_v2.csv')

# Yearly comparison
print("\n--- Yearly Returns ---")
yc = pd.DataFrame()
for name in metrics_order:
    m = all_metrics.get(name)
    if m and 'yearly_returns' in m:
        yc[name] = m['yearly_returns']
if len(yc) > 0:
    yc.index = yc.index.year
    yc.to_csv(OUT / 'yearly_returns_v2.csv', float_format='%.4f')
    print(yc.map(lambda x: f"{x:.1%}" if not pd.isna(x) else "").to_string())

# Trade log
with open(OUT / 'trade_log_v2.json', 'w') as f:
    json.dump(result_harvester.get('trade_log', []), f, indent=2)

# ============================================================
# 12. FINAL REPORT
# ============================================================
print("\n" + "=" * 70)
print("STEP 12: Final report")
print("=" * 70)

report = []
report.append("=" * 80)
report.append("ASYMMETRIC OPPORTUNITY HARVESTER v2 — BACKTEST REPORT")
report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
report.append(f"Period: {START_DATE} to {END_DATE} | Capital: ${INITIAL_CAPITAL:,}")
report.append("=" * 80)

report.append("""
DESIGN CHANGES (v1 -> v2):
  1. REMOVED UPRO leverage — Level 3 is 100% SPY, not 3x
  2. Added 5-day confirmation for backwardation (avoid buying into crash)
  3. Added 15% drawdown circuit breaker (21-day cash cooldown)
  4. Reduced min-hold from 21 to 10 days
""")

report.append("PERFORMANCE SUMMARY")
report.append("-" * 80)
header = f"  {'Metric':<25} {'SPY B&H':>14} {'Trend CTA':>14} {'Harvester':>14} {'Combined':>14}"
report.append(header)
for rn, key, fmt in metric_rows:
    vals = [fmt.format(all_metrics.get(n, {}).get(key, 0)) for n in metrics_order]
    report.append(f"  {rn:<25} {vals[0]:>14} {vals[1]:>14} {vals[2]:>14} {vals[3]:>14}")

report.append(f"\nPERMUTATION TEST: p={p_sharpe:.4f} ({'SIGNIFICANT' if p_sharpe < 0.05 else 'NOT significant'})")
report.append(f"CORRELATION Harvester-CTA: {corr_matrix.loc['Harvester', 'CTA']:.3f}")

report.append("\n" + "=" * 80)
report.append("END OF REPORT")
report.append("=" * 80)

report_text = "\n".join(report)
with open(OUT / 'backtest_report_v2.txt', 'w') as f:
    f.write(report_text)

print(report_text)
print(f"\nAll v2 outputs saved to: {OUT}")
