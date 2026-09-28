#!/usr/bin/env python3
"""
CONFLUENCE AS CONFIRMATION GATE FOR GAMEPLAN v2 — SESSION_STATE #427 FOLLOW-UP
================================================================================
Pure confluence beats baseline (Sharpe 3.34 vs 1.81) but switches 35x/yr.
Adaptive vol is modest (+0.20 Sharpe) with 10 switches/yr.

THIS TEST: Use confluence as a CONFIRMATION filter within Gameplan v2.
- Only enter UPRO when BOTH vol<threshold AND confluence_score >= gate
- If vol says UPRO but confluence says no → stay in SPY (safer default)
- If vol says SPY/GLD → follow vol regardless of confluence (crisis protection)

This should REDUCE whipsaws (fewer entries) while capturing confluence's timing edge.

Also test: confluence as EXIT signal (faster exit from UPRO when score drops).
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/confluence_gate")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100
TX_COST_PCT = 0.0002
GAP_RISK_PCT = 0.001

print("=" * 80)
print("CONFLUENCE CONFIRMATION GATE — GAMEPLAN v2 ENHANCEMENT")
print("=" * 80)

# ── DATA ──
print("\n[1/7] Downloading data...")
tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
data = yf.download(tickers, start='2012-01-01', end='2026-07-17',
                   auto_adjust=True, threads=True, progress=False)
if isinstance(data.columns, pd.MultiIndex):
    closes = data['Close']
else:
    closes = data
if hasattr(closes.columns, 'droplevel'):
    try:
        closes.columns = closes.columns.droplevel(1)
    except:
        pass
closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
returns = closes.pct_change().fillna(0)
print(f"  {len(closes)} trading days")

# ── SIGNALS ──
print("\n[2/7] Computing signals...")
spy = closes['SPY']
spy_ret = spy.pct_change()

sig = {}
sig['mom_5d'] = spy.pct_change(5)
delta = spy_ret.copy()
gain = delta.where(delta > 0, 0).rolling(10).mean()
loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
rs = gain / loss.replace(0, np.nan)
sig['rsi_10'] = 100 - (100 / (1 + rs))
sig['sma_20'] = spy.rolling(20).mean()
sig['sma_50'] = spy.rolling(50).mean()
sig['vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252) * 100
sig['sma_200'] = spy.rolling(200).mean()
sig['sma_200_slope'] = sig['sma_200'].pct_change(20)
sig['vol_63d'] = spy_ret.rolling(63).std() * np.sqrt(252) * 100
sig['vol_63d_trend'] = sig['vol_63d'] - sig['vol_63d'].rolling(21).mean()


def confluence_score(i):
    """Compute 0-3 composite score."""
    s = 0.0
    mom = sig['mom_5d'].iloc[i]
    rsi = sig['rsi_10'].iloc[i]
    s20 = sig['sma_20'].iloc[i]
    s50 = sig['sma_50'].iloc[i]
    vol21 = sig['vol_21d'].iloc[i]
    slope = sig['sma_200_slope'].iloc[i]
    vol_trend = sig['vol_63d_trend'].iloc[i]

    if not np.isnan(mom) and mom > 0: s += 0.5
    if not np.isnan(rsi) and rsi > 50: s += 0.5
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50: s += 0.5
    if not np.isnan(vol21) and vol21 < 15: s += 0.5
    if not np.isnan(slope) and slope > 0: s += 0.5
    if not np.isnan(vol_trend) and vol_trend < 0: s += 0.5
    return s


# ── SIMULATION ──
def calc_metrics(vals, contributed, switches, label=""):
    rets = pd.Series(vals).pct_change().dropna()
    if len(rets) < 2 or vals[-1] <= 0:
        return {}
    n_yr = len(vals) / 252
    cagr = (vals[-1] / vals[0]) ** (1/n_yr) - 1 if n_yr > 0 else 0
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    down = rets[rets < 0]
    sortino = rets.mean() / down.std() * np.sqrt(252) if len(down) > 0 and down.std() > 0 else 0
    cummax = pd.Series(vals).cummax()
    dd = (pd.Series(vals) - cummax) / cummax
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0
    return {
        'label': label, 'final': vals[-1], 'contributed': contributed,
        'cagr': cagr*100, 'sharpe': sharpe, 'sortino': sortino,
        'maxdd': maxdd*100, 'calmar': calmar, 'switches': switches,
        'sw_yr': switches/n_yr if n_yr > 0 else 0,
        'wr': (rets > 0).mean()
    }


def simulate(start, end, regime_fn, label=""):
    mask = (closes.index >= start) & (closes.index <= end)
    dates = closes.index[mask]
    if len(dates) < 10: return None, None

    cash = float(INITIAL)
    contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0
    vals = []
    regimes = []

    for date in dates:
        i = closes.index.get_loc(date)
        wk = (date.year, date.isocalendar()[1])
        if wk != last_week:
            cash += WEEKLY_DCA
            contributed += WEEKLY_DCA
            last_week = wk

        if i < 210:
            vals.append(cash)
            last_regime = 'CASH'
            regimes.append('CASH')
            continue

        regime = regime_fn(i, date)
        regimes.append(regime)

        if regime != last_regime and last_regime is not None and last_regime != 'CASH':
            switches += 1
            cash *= (1 - TX_COST_PCT - GAP_RISK_PCT)
        last_regime = regime

        if regime in returns.columns:
            r = returns.loc[date, regime]
            if not np.isnan(r):
                cash *= (1 + r)
        vals.append(cash)

    v = pd.Series(vals, index=dates[:len(vals)])
    m = calc_metrics(vals, contributed, switches, label)
    return v, m, regimes


# ── REGIME FUNCTIONS ──

def baseline_v2(i, date):
    if date.month == 9: return 'SPY'
    vol = sig['vol_21d'].iloc[i]
    s20 = sig['sma_20'].iloc[i]
    s200 = sig['sma_200'].iloc[i]
    if np.isnan(vol): vol = 15
    prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)
    if vol > 30: return 'GLD'
    elif vol > 15 or prot_off: return 'SPY'
    else: return 'UPRO'


def confirmation_gate(entry_gate, exit_gate=None):
    """
    Only enter UPRO when vol<15 AND confluence >= entry_gate.
    Optionally exit UPRO faster when confluence drops below exit_gate.
    Vol-based crisis protection still overrides.
    """
    def _fn(i, date):
        if date.month == 9: return 'SPY'
        vol = sig['vol_21d'].iloc[i]
        s20 = sig['sma_20'].iloc[i]
        s200 = sig['sma_200'].iloc[i]
        if np.isnan(vol): vol = 15
        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)

        # Crisis protection always wins
        if vol > 30: return 'GLD'
        if vol > 15 or prot_off: return 'SPY'

        # Vol says UPRO is OK. Check confluence gate.
        score = confluence_score(i)
        if score >= entry_gate:
            return 'UPRO'
        else:
            return 'SPY'  # Safe default when confluence disagrees
    return _fn


def dual_gate(entry_gate, exit_gate):
    """
    Entry: need vol<15 AND confluence >= entry_gate.
    Exit: drop to SPY when confluence < exit_gate (even if vol still OK).
    Creates hysteresis to reduce whipsaw.
    """
    state = {'in_upro': False}

    def _fn(i, date):
        if date.month == 9:
            state['in_upro'] = False
            return 'SPY'
        vol = sig['vol_21d'].iloc[i]
        s20 = sig['sma_20'].iloc[i]
        s200 = sig['sma_200'].iloc[i]
        if np.isnan(vol): vol = 15
        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)

        if vol > 30:
            state['in_upro'] = False
            return 'GLD'
        if vol > 15 or prot_off:
            state['in_upro'] = False
            return 'SPY'

        score = confluence_score(i)
        if state['in_upro']:
            # Already in UPRO — only exit on low score
            if score < exit_gate:
                state['in_upro'] = False
                return 'SPY'
            return 'UPRO'
        else:
            # Not in UPRO — need high score to enter
            if score >= entry_gate:
                state['in_upro'] = True
                return 'UPRO'
            return 'SPY'
    return _fn


# ══════════════════════════════════════════════════════════════════════
# FULL-PERIOD COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n[3/7] Full-period comparison...")

s, e = closes.index[0], closes.index[-1]

v_base, m_base, r_base = simulate(s, e, baseline_v2, "Baseline (v2)")
print(f"\n  {'Config':<35} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'SW/yr':>8} {'Final$':>12}")
print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12}")
print(f"  {'Baseline':<35} {m_base['sharpe']:>8.3f} {m_base['cagr']:>8.1f} {m_base['maxdd']:>8.1f} {m_base['sw_yr']:>8.1f} {m_base['final']:>12,.0f}")

configs = []

# Confirmation gate variants
for gate in [1.0, 1.5, 2.0, 2.5]:
    lbl = f"Confirm_gate_{gate:.1f}"
    v, m, r = simulate(s, e, confirmation_gate(gate), lbl)
    if m:
        configs.append(m)
        d = m['sharpe'] - m_base['sharpe']
        print(f"  {lbl:<35} {m['sharpe']:>8.3f} {m['cagr']:>8.1f} {m['maxdd']:>8.1f} {m['sw_yr']:>8.1f} {m['final']:>12,.0f}  ({d:+.3f})")

# Dual gate (hysteresis) variants
for entry_g, exit_g in [(2.0, 1.0), (2.5, 1.5), (2.0, 1.5), (2.5, 2.0), (1.5, 0.5), (1.5, 1.0)]:
    lbl = f"Dual_{entry_g:.1f}/{exit_g:.1f}"
    v, m, r = simulate(s, e, dual_gate(entry_g, exit_g), lbl)
    if m:
        configs.append(m)
        d = m['sharpe'] - m_base['sharpe']
        print(f"  {lbl:<35} {m['sharpe']:>8.3f} {m['cagr']:>8.1f} {m['maxdd']:>8.1f} {m['sw_yr']:>8.1f} {m['final']:>12,.0f}  ({d:+.3f})")


# ══════════════════════════════════════════════════════════════════════
# FIND BEST CONFIGS
# ══════════════════════════════════════════════════════════════════════
print("\n[4/7] Identifying best configs...")

# Best by Sharpe
best_confirm = max([c for c in configs if c['label'].startswith('Confirm')], key=lambda x: x['sharpe'])
best_dual = max([c for c in configs if c['label'].startswith('Dual')], key=lambda x: x['sharpe'])

# Best practical (good Sharpe + <15 switches/yr)
practical = [c for c in configs if c['sw_yr'] < 15 and c['sharpe'] > m_base['sharpe']]
best_practical = max(practical, key=lambda x: x['sharpe']) if practical else None

print(f"  Best confirmation gate: {best_confirm['label']} (Sharpe {best_confirm['sharpe']:.3f})")
print(f"  Best dual gate: {best_dual['label']} (Sharpe {best_dual['sharpe']:.3f})")
if best_practical:
    print(f"  Best practical (<15 sw/yr): {best_practical['label']} (Sharpe {best_practical['sharpe']:.3f}, {best_practical['sw_yr']:.1f} sw/yr)")


# ══════════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════
print("\n[5/7] Walk-forward validation...")

def walk_forward(regime_fn, label):
    wf = []
    for yr in range(2014, 2025):
        v, m, _ = simulate(f"{yr}-01-01", f"{yr}-12-31", regime_fn, f"WF_{yr}")
        if m:
            wf.append({'year': yr, 'sharpe': m['sharpe'], 'cagr': m['cagr'], 'maxdd': m['maxdd']})
    if wf:
        sharpes = [w['sharpe'] for w in wf]
        pos = sum(1 for s in sharpes if s > 0)
        mean_s = np.mean(sharpes)
        print(f"  {label}: WF mean Sharpe {mean_s:.3f}, {pos}/{len(sharpes)} positive")
        return wf, mean_s
    return [], 0

wf_base, wf_base_s = walk_forward(baseline_v2, "Baseline")

# Walk-forward the best practical config
if best_practical:
    lbl = best_practical['label']
    if lbl.startswith('Confirm'):
        gate = float(lbl.split('_')[-1])
        wf_best, wf_best_s = walk_forward(confirmation_gate(gate), lbl)
    elif lbl.startswith('Dual'):
        parts = lbl.replace('Dual_', '').split('/')
        eg, xg = float(parts[0]), float(parts[1])
        wf_best, wf_best_s = walk_forward(dual_gate(eg, xg), lbl)

# Also walk-forward best absolute Sharpe configs
if best_confirm['label'] != (best_practical['label'] if best_practical else ''):
    lbl = best_confirm['label']
    gate = float(lbl.split('_')[-1])
    wf_confirm, wf_confirm_s = walk_forward(confirmation_gate(gate), lbl)

if best_dual['label'] != (best_practical['label'] if best_practical else ''):
    lbl = best_dual['label']
    parts = lbl.replace('Dual_', '').split('/')
    eg, xg = float(parts[0]), float(parts[1])
    wf_dual, wf_dual_s = walk_forward(dual_gate(eg, xg), lbl)


# ══════════════════════════════════════════════════════════════════════
# PERMUTATION TEST
# ══════════════════════════════════════════════════════════════════════
print("\n[6/7] Permutation test (200 shuffles on best practical)...")

def permutation_test(regime_fn, real_sharpe, label, n=200):
    mask = closes.index >= closes.index[210]
    dates = closes.index[mask]
    real_regimes = [regime_fn(closes.index.get_loc(d), d) for d in dates]
    regime_set = list(set(real_regimes))
    dist = [real_regimes.count(r)/len(real_regimes) for r in regime_set]

    perm_sharpes = []
    for p in range(n):
        np.random.seed(p + 2000)
        rand_reg = np.random.choice(regime_set, size=len(dates), p=dist)

        cash = float(INITIAL)
        contributed = float(INITIAL)
        last_wk = None
        last_r = None
        vals = []
        for j, date in enumerate(dates):
            wk = (date.year, date.isocalendar()[1])
            if wk != last_wk:
                cash += WEEKLY_DCA
                contributed += WEEKLY_DCA
                last_wk = wk
            r = rand_reg[j]
            if r != last_r and last_r is not None:
                cash *= (1 - TX_COST_PCT - GAP_RISK_PCT)
            last_r = r
            if r in returns.columns:
                ret = returns.loc[date, r]
                if not np.isnan(ret):
                    cash *= (1 + ret)
            vals.append(cash)

        rets = pd.Series(vals).pct_change().dropna()
        if rets.std() > 0:
            perm_sharpes.append(rets.mean() / rets.std() * np.sqrt(252))

    beat = sum(1 for s in perm_sharpes if s >= real_sharpe)
    p_val = beat / len(perm_sharpes) if perm_sharpes else 1.0
    print(f"  {label}: p={p_val:.3f} (real {real_sharpe:.3f} vs perm mean {np.mean(perm_sharpes):.3f})")
    return p_val

if best_practical:
    lbl = best_practical['label']
    if lbl.startswith('Confirm'):
        gate = float(lbl.split('_')[-1])
        perm_p = permutation_test(confirmation_gate(gate), best_practical['sharpe'], lbl)
    elif lbl.startswith('Dual'):
        parts = lbl.replace('Dual_', '').split('/')
        eg, xg = float(parts[0]), float(parts[1])
        perm_p = permutation_test(dual_gate(eg, xg), best_practical['sharpe'], lbl)

perm_base = permutation_test(baseline_v2, m_base['sharpe'], "Baseline")


# ══════════════════════════════════════════════════════════════════════
# R1 + SUB-PERIOD + YEAR-BY-YEAR
# ══════════════════════════════════════════════════════════════════════
print("\n[7/7] R1, sub-period, year-by-year...")

def r1_test(regime_fn, label):
    spy_ret = returns['SPY']
    mask = closes.index >= closes.index[210]
    dates = closes.index[mask]

    rets_by_day = {'green': [], 'red': []}
    for date in dates:
        i = closes.index.get_loc(date)
        regime = regime_fn(i, date)
        r = returns.loc[date, regime] if regime in returns.columns else 0
        sr = spy_ret.loc[date]
        if sr > 0.001: rets_by_day['green'].append(r)
        elif sr < -0.001: rets_by_day['red'].append(r)

    sg = np.mean(rets_by_day['green']) / np.std(rets_by_day['green']) * np.sqrt(252) if len(rets_by_day['green']) > 10 else 0
    sr = np.mean(rets_by_day['red']) / np.std(rets_by_day['red']) * np.sqrt(252) if len(rets_by_day['red']) > 10 else 0
    gap = abs(sg - sr) / max(abs(sg), abs(sr), 1e-6)
    print(f"  {label}: R1 gap {gap:.3f} {'PASS' if gap <= 0.5 else 'FAIL'} (green {sg:.2f}, red {sr:.2f})")
    return gap

if best_practical:
    lbl = best_practical['label']
    if lbl.startswith('Confirm'):
        gate = float(lbl.split('_')[-1])
        gap_best = r1_test(confirmation_gate(gate), lbl)
    elif lbl.startswith('Dual'):
        parts = lbl.replace('Dual_', '').split('/')
        eg, xg = float(parts[0]), float(parts[1])
        gap_best = r1_test(dual_gate(eg, xg), lbl)

gap_base = r1_test(baseline_v2, "Baseline")


# Sub-period
def sub_period(regime_fn, label):
    periods = [('2013-01-01','2017-12-31'), ('2018-01-01','2021-12-31'), ('2022-01-01','2025-12-31')]
    sharpes = []
    for s_d, e_d in periods:
        v, m, _ = simulate(s_d, e_d, regime_fn, f"sub")
        if m: sharpes.append(m['sharpe'])
    cv = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else 999
    print(f"  {label}: Sub Sharpes {[f'{s:.2f}' for s in sharpes]}, CV {cv:.3f}")
    return cv

if best_practical:
    lbl = best_practical['label']
    if lbl.startswith('Confirm'):
        gate = float(lbl.split('_')[-1])
        cv_best = sub_period(confirmation_gate(gate), lbl)
    elif lbl.startswith('Dual'):
        parts = lbl.replace('Dual_', '').split('/')
        eg, xg = float(parts[0]), float(parts[1])
        cv_best = sub_period(dual_gate(eg, xg), lbl)

cv_base = sub_period(baseline_v2, "Baseline")


# Year-by-year comparison
print("\n  Year-by-year (best practical vs baseline):")
if best_practical:
    lbl = best_practical['label']
    print(f"  {'Year':>6} {'Baseline':>10} {'Best':>10} {'Delta':>10}")
    for yr in range(2013, 2026):
        v_b, m_b, _ = simulate(f"{yr}-01-01", f"{yr}-12-31", baseline_v2, "b")
        if lbl.startswith('Confirm'):
            gate = float(lbl.split('_')[-1])
            v_t, m_t, _ = simulate(f"{yr}-01-01", f"{yr}-12-31", confirmation_gate(gate), "t")
        elif lbl.startswith('Dual'):
            parts = lbl.replace('Dual_', '').split('/')
            eg, xg = float(parts[0]), float(parts[1])
            v_t, m_t, _ = simulate(f"{yr}-01-01", f"{yr}-12-31", dual_gate(eg, xg), "t")
        if m_b and m_t:
            delta = m_t['sharpe'] - m_b['sharpe']
            marker = "✓" if delta > 0 else "✗"
            print(f"  {yr:>6} {m_b['sharpe']:>10.2f} {m_t['sharpe']:>10.2f} {delta:>+10.2f} {marker}")


# ══════════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

print(f"\n  Baseline: Sharpe {m_base['sharpe']:.3f}, CAGR {m_base['cagr']:.1f}%, MaxDD {m_base['maxdd']:.1f}%, {m_base['sw_yr']:.1f} sw/yr")
if best_practical:
    m = best_practical
    d = m['sharpe'] - m_base['sharpe']
    print(f"  Best practical: {m['label']} — Sharpe {m['sharpe']:.3f} ({d:+.3f}), "
          f"CAGR {m['cagr']:.1f}%, MaxDD {m['maxdd']:.1f}%, {m['sw_yr']:.1f} sw/yr")

    if d > 0.1:
        print(f"\n  ✅ CONFLUENCE CONFIRMATION GATE IMPROVES Gameplan v2")
        print(f"     {d:+.3f} Sharpe improvement with manageable switching")
        print(f"     RECOMMENDATION: Integrate into Gameplan v2 as entry confirmation")
    elif d > 0:
        print(f"\n  🟡 MARGINAL IMPROVEMENT ({d:+.3f} Sharpe)")
        print(f"     Not worth the added complexity")
    else:
        print(f"\n  ❌ NO IMPROVEMENT — confluence gate does not help within Gameplan v2")
        print(f"     The timing benefit only works as a standalone regime detector")

# Save
results = {
    'baseline': m_base,
    'configs': configs,
    'best_practical': best_practical,
    'best_confirm': best_confirm,
    'best_dual': best_dual,
}
with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n  Results saved to {OUTPUT_DIR}")
print("=" * 80)
