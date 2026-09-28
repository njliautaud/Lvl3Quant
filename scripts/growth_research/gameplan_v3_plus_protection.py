#!/usr/bin/env python3
"""
GAMEPLAN v3 + 4-SIGNAL PROTECTION OVERLAY — COMBINED SYSTEM TEST
=================================================================
Entry 428: Confluence gate (Dual 2.5/2.0) → Sharpe 2.388, MaxDD -25.2%
Entry 355/360: 4-signal protection → Sharpe 3.24, MaxDD -9.1%

These use DIFFERENT signal families:
  v3 confluence: momentum, RSI, MA cross, vol level, 200d slope, vol trend
  Protection: VIX<20, SPY>50SMA, credit spread, market breadth

Hypothesis: combining both should capture BOTH benefits:
  - Confluence prevents entering UPRO during low-vol breakdowns
  - Protection overlay catches macro risk (credit, breadth, VIX) that trend signals miss

Also test: are they redundant (already overlapping) or complementary?
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/v3_plus_protection")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100
TX_COST_PCT = 0.0002
GAP_RISK_PCT = 0.001

print("=" * 80)
print("GAMEPLAN v3 + 4-SIGNAL PROTECTION — COMBINED SYSTEM")
print("=" * 80)

# ── DATA ──
print("\n[1/8] Downloading data...")
tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'QQQ', 'HYG', 'LQD', 'IWM', 'VIXY']
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
print(f"  {len(closes)} trading days: {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")

# ── SIGNALS ──
print("\n[2/8] Computing all signals...")
spy = closes['SPY']
spy_ret = spy.pct_change()

# Confluence signals
mom_5d = spy.pct_change(5)
delta = spy_ret.copy()
gain = delta.where(delta > 0, 0).rolling(10).mean()
loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
rs = gain / loss.replace(0, np.nan)
rsi_10 = 100 - (100 / (1 + rs))
sma_20 = spy.rolling(20).mean()
sma_50 = spy.rolling(50).mean()
vol_21d = spy_ret.rolling(21).std() * np.sqrt(252) * 100
sma_200 = spy.rolling(200).mean()
sma_200_slope = sma_200.pct_change(20)
vol_63d = spy_ret.rolling(63).std() * np.sqrt(252) * 100
vol_63d_trend = vol_63d - vol_63d.rolling(21).mean()

# Protection overlay signals
# VIX proxy from VIXY
vixy_sma10 = closes['VIXY'].rolling(10).mean() if 'VIXY' in closes.columns else None

# Credit spread: LQD/HYG ratio (rising = stress)
credit_ratio = closes['LQD'] / closes['HYG'] if 'LQD' in closes.columns and 'HYG' in closes.columns else None
credit_sma20 = credit_ratio.rolling(20).mean() if credit_ratio is not None else None

# Breadth: IWM relative to SPY (small caps leading = healthy)
breadth_21d = (closes['IWM'].pct_change(21) - spy.pct_change(21)) if 'IWM' in closes.columns else None

# QQQ above 50SMA
qqq_sma50 = closes['QQQ'].rolling(50).mean() if 'QQQ' in closes.columns else None

print("  All signals computed.")


def confluence_score(i):
    s = 0.0
    m = mom_5d.iloc[i]
    r = rsi_10.iloc[i]
    s20 = sma_20.iloc[i]
    s50 = sma_50.iloc[i]
    v21 = vol_21d.iloc[i]
    slope = sma_200_slope.iloc[i]
    vt = vol_63d_trend.iloc[i]

    if not np.isnan(m) and m > 0: s += 0.5
    if not np.isnan(r) and r > 50: s += 0.5
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50: s += 0.5
    if not np.isnan(v21) and v21 < 15: s += 0.5
    if not np.isnan(slope) and slope > 0: s += 0.5
    if not np.isnan(vt) and vt < 0: s += 0.5
    return s


def protection_signals(i, date):
    """4-signal protection overlay (entry 355). Returns (passes, n_signals_met)."""
    n_met = 0

    # 1. Vol < 20% (using 21d realized vol as VIX proxy)
    v = vol_21d.iloc[i]
    if not np.isnan(v) and v < 20:
        n_met += 1

    # 2. SPY > 50SMA
    s50 = sma_50.iloc[i]
    if not np.isnan(s50) and spy.iloc[i] > s50:
        n_met += 1

    # 3. Credit not stressed (LQD/HYG ratio below 20d SMA = no stress)
    if credit_ratio is not None and credit_sma20 is not None:
        cr = credit_ratio.iloc[i]
        cs = credit_sma20.iloc[i]
        if not np.isnan(cr) and not np.isnan(cs) and cr <= cs:
            n_met += 1
    else:
        n_met += 1  # If no data, assume OK

    # 4. Breadth > -3% (small caps not lagging badly)
    if breadth_21d is not None:
        b = breadth_21d.iloc[i]
        if not np.isnan(b) and b > -0.03:
            n_met += 1
    else:
        n_met += 1

    return n_met >= 4, n_met  # All 4 must pass


def protection_relaxed(i, date):
    """3-of-4 protection (more lenient)."""
    _, n = protection_signals(i, date)
    return n >= 3, n


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
        'wr': (rets > 0).mean(), 'n_years': n_yr
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
            continue

        regime = regime_fn(i, date)

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
    return v, m


# ── REGIME FUNCTIONS ──

def baseline_v2(i, date):
    if date.month == 9: return 'SPY'
    v = vol_21d.iloc[i]
    s20 = sma_20.iloc[i]
    s200 = sma_200.iloc[i]
    if np.isnan(v): v = 15
    prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)
    if v > 30: return 'GLD'
    elif v > 15 or prot_off: return 'SPY'
    else: return 'UPRO'


class DualGateV3:
    """Gameplan v3: confluence confirmation gate with hysteresis."""
    def __init__(self, entry=2.5, exit=2.0):
        self.entry = entry
        self.exit = exit
        self.in_upro = False

    def __call__(self, i, date):
        if date.month == 9:
            self.in_upro = False
            return 'SPY'
        v = vol_21d.iloc[i]
        s20 = sma_20.iloc[i]
        s200 = sma_200.iloc[i]
        if np.isnan(v): v = 15
        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)
        if v > 30:
            self.in_upro = False
            return 'GLD'
        if v > 15 or prot_off:
            self.in_upro = False
            return 'SPY'

        score = confluence_score(i)
        if self.in_upro:
            if score < self.exit:
                self.in_upro = False
                return 'SPY'
            return 'UPRO'
        else:
            if score >= self.entry:
                self.in_upro = True
                return 'UPRO'
            return 'SPY'


class ProtectionOnly:
    """4-signal protection overlay on top of v2."""
    def __init__(self, require_all=True):
        self.require_all = require_all

    def __call__(self, i, date):
        base = baseline_v2(i, date)
        if base != 'UPRO':
            return base  # Vol/MA already says no UPRO

        # Additional protection check
        if self.require_all:
            passes, _ = protection_signals(i, date)
        else:
            passes, _ = protection_relaxed(i, date)

        return 'UPRO' if passes else 'SPY'


class CombinedV3Protection:
    """v3 confluence gate + protection overlay. Both must agree for UPRO."""
    def __init__(self, entry=2.5, exit_g=2.0, require_all_prot=True):
        self.entry = entry
        self.exit_g = exit_g
        self.in_upro = False
        self.require_all = require_all_prot

    def __call__(self, i, date):
        if date.month == 9:
            self.in_upro = False
            return 'SPY'
        v = vol_21d.iloc[i]
        s20 = sma_20.iloc[i]
        s200 = sma_200.iloc[i]
        if np.isnan(v): v = 15
        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)
        if v > 30:
            self.in_upro = False
            return 'GLD'
        if v > 15 or prot_off:
            self.in_upro = False
            return 'SPY'

        # Confluence gate
        score = confluence_score(i)
        confluence_ok = (score >= self.exit_g) if self.in_upro else (score >= self.entry)

        # Protection overlay
        if self.require_all:
            prot_ok, _ = protection_signals(i, date)
        else:
            prot_ok, _ = protection_relaxed(i, date)

        if confluence_ok and prot_ok:
            self.in_upro = True
            return 'UPRO'
        else:
            self.in_upro = False
            return 'SPY'


class EitherGate:
    """UPRO if EITHER confluence OR protection passes (more aggressive)."""
    def __init__(self, entry=2.5, exit_g=2.0):
        self.entry = entry
        self.exit_g = exit_g
        self.in_upro = False

    def __call__(self, i, date):
        if date.month == 9:
            self.in_upro = False
            return 'SPY'
        v = vol_21d.iloc[i]
        s20 = sma_20.iloc[i]
        s200 = sma_200.iloc[i]
        if np.isnan(v): v = 15
        prot_off = (not np.isnan(s20) and not np.isnan(s200) and s20 < s200)
        if v > 30:
            self.in_upro = False
            return 'GLD'
        if v > 15 or prot_off:
            self.in_upro = False
            return 'SPY'

        score = confluence_score(i)
        confluence_ok = (score >= self.exit_g) if self.in_upro else (score >= self.entry)
        prot_ok, _ = protection_signals(i, date)

        if confluence_ok or prot_ok:
            self.in_upro = True
            return 'UPRO'
        else:
            self.in_upro = False
            return 'SPY'


# ══════════════════════════════════════════════════════════════════════
# FULL-PERIOD COMPARISON
# ══════════════════════════════════════════════════════════════════════
print("\n[3/8] Full-period comparison...")

s, e = closes.index[0], closes.index[-1]

configs = [
    ("Baseline (v2)", baseline_v2),
    ("v3 Confluence Gate", DualGateV3(2.5, 2.0)),
    ("Protection 4/4", ProtectionOnly(require_all=True)),
    ("Protection 3/4", ProtectionOnly(require_all=False)),
    ("Combined (v3+prot 4/4)", CombinedV3Protection(2.5, 2.0, True)),
    ("Combined (v3+prot 3/4)", CombinedV3Protection(2.5, 2.0, False)),
    ("Either (v3 OR prot)", EitherGate(2.5, 2.0)),
]

results = []
print(f"\n  {'Config':<30} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'SW/yr':>8} {'Final$':>12} {'Calmar':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*12} {'-'*8}")

for lbl, fn in configs:
    v, m = simulate(s, e, fn, lbl)
    if m:
        results.append(m)
        print(f"  {lbl:<30} {m['sharpe']:>8.3f} {m['cagr']:>8.1f} {m['maxdd']:>8.1f} {m['sw_yr']:>8.1f} {m['final']:>12,.0f} {m['calmar']:>8.2f}")


# ══════════════════════════════════════════════════════════════════════
# WALK-FORWARD
# ══════════════════════════════════════════════════════════════════════
print("\n[4/8] Walk-forward (11 windows, 2014-2024)...")

def walk_forward(regime_fn_factory, label):
    """regime_fn_factory: callable that returns a fresh regime_fn."""
    sharpes = []
    for yr in range(2014, 2025):
        fn = regime_fn_factory()
        v, m = simulate(f"{yr}-01-01", f"{yr}-12-31", fn, f"WF_{yr}")
        if m:
            sharpes.append(m['sharpe'])
    pos = sum(1 for s in sharpes if s > 0)
    mean_s = np.mean(sharpes) if sharpes else 0
    print(f"  {label:<30}: WF mean Sharpe {mean_s:.3f}, {pos}/{len(sharpes)} positive")
    return sharpes, mean_s

wf_configs = [
    ("Baseline (v2)", lambda: baseline_v2),
    ("v3 Confluence Gate", lambda: DualGateV3(2.5, 2.0)),
    ("Protection 4/4", lambda: ProtectionOnly(True)),
    ("Combined (v3+prot 4/4)", lambda: CombinedV3Protection(2.5, 2.0, True)),
    ("Combined (v3+prot 3/4)", lambda: CombinedV3Protection(2.5, 2.0, False)),
    ("Either (v3 OR prot)", lambda: EitherGate(2.5, 2.0)),
]

wf_results = {}
for lbl, factory in wf_configs:
    sharpes, mean_s = walk_forward(factory, lbl)
    wf_results[lbl] = {'sharpes': sharpes, 'mean': mean_s}


# ══════════════════════════════════════════════════════════════════════
# PERMUTATION TEST (best combined config)
# ══════════════════════════════════════════════════════════════════════
print("\n[5/8] Permutation test (200 shuffles)...")

def permutation_test(regime_fn, real_sharpe, label, n=200):
    mask = closes.index >= closes.index[210]
    dates = closes.index[mask]
    real_regimes = [regime_fn(closes.index.get_loc(d), d) for d in dates]
    regime_set = list(set(real_regimes))
    dist = [real_regimes.count(r)/len(real_regimes) for r in regime_set]

    perm_sharpes = []
    for p in range(n):
        np.random.seed(p + 3000)
        rand_reg = np.random.choice(regime_set, size=len(dates), p=dist)
        cash = float(INITIAL)
        cont = float(INITIAL)
        last_wk = None
        last_r = None
        vals = []
        for j, date in enumerate(dates):
            wk = (date.year, date.isocalendar()[1])
            if wk != last_wk:
                cash += WEEKLY_DCA
                cont += WEEKLY_DCA
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

    beat = sum(1 for ps in perm_sharpes if ps >= real_sharpe)
    p_val = beat / len(perm_sharpes)
    print(f"  {label}: p={p_val:.3f} (real {real_sharpe:.3f} vs perm mean {np.mean(perm_sharpes):.3f})")
    return p_val

# Test top 3
for lbl, fn in [("Baseline", baseline_v2),
                ("v3 Gate", DualGateV3(2.5, 2.0)),
                ("Combined 4/4", CombinedV3Protection(2.5, 2.0, True)),
                ("Combined 3/4", CombinedV3Protection(2.5, 2.0, False))]:
    m = [r for r in results if r['label'].startswith(lbl[:10])][0] if any(r['label'].startswith(lbl[:10]) for r in results) else None
    if m:
        # Need fresh instance
        if lbl == "Baseline":
            permutation_test(baseline_v2, m['sharpe'], lbl)
        elif lbl == "v3 Gate":
            permutation_test(DualGateV3(2.5, 2.0), m['sharpe'], lbl)
        elif lbl == "Combined 4/4":
            permutation_test(CombinedV3Protection(2.5, 2.0, True), m['sharpe'], lbl)
        elif lbl == "Combined 3/4":
            permutation_test(CombinedV3Protection(2.5, 2.0, False), m['sharpe'], lbl)


# ══════════════════════════════════════════════════════════════════════
# R1 TEST
# ══════════════════════════════════════════════════════════════════════
print("\n[6/8] R1 regime-agnostic test...")

def r1_test(regime_fn, label):
    spy_ret = returns['SPY']
    mask = closes.index >= closes.index[210]
    dates = closes.index[mask]
    by_day = {'green': [], 'red': []}
    for date in dates:
        i = closes.index.get_loc(date)
        regime = regime_fn(i, date)
        r = returns.loc[date, regime] if regime in returns.columns else 0
        sr = spy_ret.loc[date]
        if sr > 0.001: by_day['green'].append(r)
        elif sr < -0.001: by_day['red'].append(r)

    sg = np.mean(by_day['green']) / np.std(by_day['green']) * np.sqrt(252) if len(by_day['green']) > 10 else 0
    sr = np.mean(by_day['red']) / np.std(by_day['red']) * np.sqrt(252) if len(by_day['red']) > 10 else 0
    gap = abs(sg - sr) / max(abs(sg), abs(sr), 1e-6)

    # HC #709 nuanced: for growth, check if red-day loss < 50% of green-day gain
    green_mean = np.mean(by_day['green']) if by_day['green'] else 0
    red_mean = np.mean(by_day['red']) if by_day['red'] else 0
    loss_ratio = abs(red_mean) / abs(green_mean) if green_mean != 0 else 999

    print(f"  {label:<30}: gap {gap:.3f} {'PASS' if gap <= 0.5 else 'FAIL'}, "
          f"green {sg:.2f}, red {sr:.2f}, loss_ratio {loss_ratio:.3f}")
    return gap, loss_ratio

for lbl, fn in [("Baseline", baseline_v2),
                ("v3 Gate", DualGateV3(2.5, 2.0)),
                ("Combined 4/4", CombinedV3Protection(2.5, 2.0, True)),
                ("Combined 3/4", CombinedV3Protection(2.5, 2.0, False))]:
    r1_test(fn, lbl)


# ══════════════════════════════════════════════════════════════════════
# SUB-PERIOD + OVERLAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n[7/8] Sub-period consistency...")

def sub_period(regime_fn_factory, label):
    periods = [('2013-01-01','2017-12-31'), ('2018-01-01','2021-12-31'), ('2022-01-01','2025-12-31')]
    sharpes = []
    for s_d, e_d in periods:
        fn = regime_fn_factory()
        v, m = simulate(s_d, e_d, fn, "sub")
        if m: sharpes.append(m['sharpe'])
    cv = np.std(sharpes) / np.mean(sharpes) if np.mean(sharpes) != 0 else 999
    print(f"  {label:<30}: {[f'{s:.2f}' for s in sharpes]}, CV {cv:.3f}")
    return cv

for lbl, factory in wf_configs:
    sub_period(factory, lbl)


# ══════════════════════════════════════════════════════════════════════
# OVERLAP ANALYSIS: How often do confluence and protection disagree?
# ══════════════════════════════════════════════════════════════════════
print("\n[8/8] Signal overlap analysis...")

mask = closes.index >= closes.index[210]
test_dates = closes.index[mask]

# Track when each gate would block UPRO (when vol says OK)
gate_v3 = DualGateV3(2.5, 2.0)
both_block = 0
only_conf_blocks = 0
only_prot_blocks = 0
neither_blocks = 0
total_upro_days = 0

for date in test_dates:
    i = closes.index.get_loc(date)

    # Would baseline say UPRO?
    base = baseline_v2(i, date)
    if base != 'UPRO':
        continue  # Only analyze days where vol says UPRO is OK

    total_upro_days += 1
    v3_says = gate_v3(i, date)
    prot_ok, _ = protection_signals(i, date)

    conf_blocks = (v3_says != 'UPRO')
    prot_blocks = (not prot_ok)

    if conf_blocks and prot_blocks:
        both_block += 1
    elif conf_blocks and not prot_blocks:
        only_conf_blocks += 1
    elif not conf_blocks and prot_blocks:
        only_prot_blocks += 1
    else:
        neither_blocks += 1

print(f"\n  Days where v2 says UPRO: {total_upro_days}")
print(f"  Both gates agree UPRO:         {neither_blocks:>5} ({neither_blocks/total_upro_days*100:.1f}%)")
print(f"  Only confluence blocks:        {only_conf_blocks:>5} ({only_conf_blocks/total_upro_days*100:.1f}%)")
print(f"  Only protection blocks:        {only_prot_blocks:>5} ({only_prot_blocks/total_upro_days*100:.1f}%)")
print(f"  Both block:                    {both_block:>5} ({both_block/total_upro_days*100:.1f}%)")
print(f"  Overlap coefficient:           {both_block/(only_conf_blocks+only_prot_blocks+both_block)*100:.1f}% (higher=more redundant)")


# ══════════════════════════════════════════════════════════════════════
# VERDICT
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("VERDICT")
print("=" * 80)

# Find best by Sharpe
best = max(results, key=lambda x: x['sharpe'])
practical = [r for r in results if r['sw_yr'] < 20]
best_practical = max(practical, key=lambda x: x['sharpe']) if practical else best

base = [r for r in results if 'Baseline' in r['label']][0]
v3 = [r for r in results if 'v3 Confluence' in r['label']][0]

print(f"\n  COMPARISON TABLE:")
print(f"  {'Config':<30} {'Sharpe':>8} {'CAGR%':>8} {'MaxDD%':>8} {'SW/yr':>8}")
print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for r in results:
    d = r['sharpe'] - base['sharpe']
    print(f"  {r['label']:<30} {r['sharpe']:>8.3f} {r['cagr']:>8.1f} {r['maxdd']:>8.1f} {r['sw_yr']:>8.1f}  ({d:+.3f})")

print(f"\n  BEST ABSOLUTE: {best['label']} — Sharpe {best['sharpe']:.3f}")
print(f"  BEST PRACTICAL: {best_practical['label']} — Sharpe {best_practical['sharpe']:.3f}, {best_practical['sw_yr']:.1f} sw/yr")

combined = [r for r in results if 'Combined' in r['label']]
if combined:
    best_combined = max(combined, key=lambda x: x['sharpe'])
    v3_sharpe = v3['sharpe']
    comb_sharpe = best_combined['sharpe']
    if comb_sharpe > v3_sharpe:
        print(f"\n  ✅ COMBINED SYSTEM IMPROVES OVER v3 ALONE (+{comb_sharpe-v3_sharpe:.3f} Sharpe)")
        print(f"     Protection overlay and confluence gate are COMPLEMENTARY")
    elif comb_sharpe > base['sharpe']:
        print(f"\n  🟡 COMBINED beats baseline (+{comb_sharpe-base['sharpe']:.3f}) but not v3 alone ({comb_sharpe-v3_sharpe:+.3f})")
        print(f"     Protection overlay is REDUNDANT when confluence gate is active")
    else:
        print(f"\n  ❌ COMBINED does not beat baseline — gates are too restrictive together")

# Save
with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump({'results': results, 'wf': {k: v['mean'] for k, v in wf_results.items()}}, f, indent=2, default=str)

print(f"\n  Saved to {OUTPUT_DIR}")
print("=" * 80)
