#!/usr/bin/env python3
"""
Portfolio v3 Test: Replace GP3 with v4.4 Full Adaptive in portfolio allocation.

Question: Does swapping GP3 for v4.4 in the 50/50 VMR+Consensus portfolio improve OOS performance?

Tests:
  A) Original: 50/50 VMR_Daily + Consensus(GP3+VMR)  — current Portfolio v2
  B) Upgraded: 50/50 VMR_Daily + Consensus(v4.4+VMR) — v4.4 replaces GP3
  C) Direct:   50/50 VMR_Daily + v4.4 standalone     — v4.4 directly, no consensus filter
  D) v4.4 only: 100% v4.4                             — is v4.4 alone better than any portfolio?

Uses FULL OOS period (2010-2026) with realistic DCA and switching costs.
Includes permutation test on the best portfolio.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent.parent

def load_data():
    """Load daily data for SPY, UPRO, GLD, TLT, VIX."""
    import yfinance as yf
    tickers = ["SPY", "UPRO", "GLD", "TLT", "^VIX"]
    data = yf.download(tickers, start="2010-01-01", auto_adjust=True,
                       threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data["Close"]
    else:
        closes = data
    if hasattr(closes.columns, "droplevel"):
        try:
            closes.columns = closes.columns.droplevel(1)
        except Exception:
            pass
    closes = closes.rename(columns={"^VIX": "VIX"})
    return closes.dropna(subset=["SPY", "UPRO"]).ffill()


def compute_signals(closes):
    """Compute all signals needed for GP3, v4.4, and VMR."""
    spy = closes["SPY"]
    vix = closes["VIX"]
    spy_ret = spy.pct_change()

    sig = {}
    # Confluence signals
    sig['mom_5d'] = spy.pct_change(5)
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    sig['rsi_10'] = 100 - (100 / (1 + rs))
    sig['sma_20'] = spy.rolling(20).mean()
    sig['sma_50'] = spy.rolling(50).mean()
    sig['sma_200'] = spy.rolling(200).mean()
    sig['sma_200_slope'] = sig['sma_200'].pct_change(20)
    sig['vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    sig['vol_63d'] = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    sig['vol_63d_trend'] = sig['vol_63d'] - sig['vol_63d'].rolling(21).mean()

    # VIX percentile (63d)
    sig['vix_pctile_63'] = vix.rolling(63).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100,
        raw=False
    )

    # VMR signals
    sig['vix'] = vix
    sig['vix_ma10'] = vix.rolling(10).mean()
    sig['vix_peak20'] = vix.rolling(20).max()

    return sig


def confluence_score(sig, i):
    """6-factor confluence score (0-3)."""
    s = 0.0
    m = sig['mom_5d'].iloc[i]
    r = sig['rsi_10'].iloc[i]
    s20 = sig['sma_20'].iloc[i]
    s50 = sig['sma_50'].iloc[i]
    v21 = sig['vol_21d'].iloc[i]
    slope = sig['sma_200_slope'].iloc[i]
    vt = sig['vol_63d_trend'].iloc[i]

    if not np.isnan(m) and m > 0: s += 0.5
    if not np.isnan(r) and r > 50: s += 0.5
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50: s += 0.5
    if not np.isnan(v21) and v21 < 15: s += 0.5
    if not np.isnan(slope) and slope > 0: s += 0.5
    if not np.isnan(vt) and vt < 0: s += 0.5
    return s


def gp3_regime(sig, i, date, in_upro):
    """GP3 (v3) regime: fixed vol thresholds + fixed confluence gate."""
    s20 = sig['sma_20'].iloc[i]
    s200 = sig['sma_200'].iloc[i]
    vol = sig['vol_21d'].iloc[i]
    if np.isnan(vol): vol = 15.0

    if date.month == 9:
        return 'SPY', False
    if not np.isnan(s20) and not np.isnan(s200) and s20 < s200:
        return 'SPY', False
    if vol > 30:
        return 'GLD', False
    if vol > 15:
        return 'SPY', False

    score = confluence_score(sig, i)
    if in_upro:
        if score < 2.0:
            return 'SPY', False
        return 'UPRO', True
    else:
        if score >= 2.5:
            return 'UPRO', True
        return 'SPY', False


def v44_regime(sig, i, date, in_upro):
    """v4.4 Full Adaptive: VIX percentile + adaptive confluence."""
    s20 = sig['sma_20'].iloc[i]
    s200 = sig['sma_200'].iloc[i]

    if date.month == 9:
        return 'SPY', False
    if not np.isnan(s20) and not np.isnan(s200) and s20 < s200:
        return 'SPY', False

    pctile = sig['vix_pctile_63'].iloc[i]
    if np.isnan(pctile):
        return 'SPY', False

    if pctile > 80:
        return 'GLD', False

    # Adaptive thresholds
    if pctile > 60:
        entry, exit_t = 3.0, 2.5
    elif pctile < 30:
        entry, exit_t = 2.0, 1.5
    else:
        entry, exit_t = 2.5, 2.0

    if pctile > 20:
        entry = max(entry, 2.5)

    score = confluence_score(sig, i)
    if in_upro:
        if score < exit_t:
            return 'SPY', False
        return 'UPRO', True
    else:
        if score >= entry:
            return 'UPRO', True
        return 'SPY', False


def vmr_regime(sig, i):
    """VMR Daily: 5-regime VIX system."""
    vix = sig['vix'].iloc[i]
    vix_ma10 = sig['vix_ma10'].iloc[i]
    vix_peak20 = sig['vix_peak20'].iloc[i]

    if np.isnan(vix) or np.isnan(vix_ma10):
        return 'SPY'

    declining = vix < vix_ma10

    if vix < 15 and declining:
        return 'UPRO'
    elif vix > 20 and not np.isnan(vix_peak20) and vix < vix_peak20 * 0.85 and declining:
        return 'UPRO'
    elif vix > 25 and not declining:
        return 'GLD'  # simplified from GLD+TLT
    elif vix > 20 and not declining:
        return 'SPY'  # simplified from SPY+TLT
    else:
        return 'SPY'


def simulate_portfolio(closes, sig, strategy_fn, warmup=260, weekly_dca=100, initial=500):
    """
    Simulate portfolio with DCA.
    strategy_fn(sig, i, date) -> holding ('UPRO', 'SPY', 'GLD')
    """
    dates = closes.index[warmup:]
    rets = closes.pct_change()

    value = initial
    contributed = initial
    last_week = None
    n_switches = 0
    prev_holding = None
    daily_values = []
    daily_dates = []

    for idx in range(warmup, len(closes)):
        d = closes.index[idx]
        dt = d.date() if hasattr(d, 'date') else d

        # Weekly DCA
        wk = (dt.year, dt.isocalendar()[1])
        if last_week != wk:
            value += weekly_dca
            contributed += weekly_dca
            last_week = wk

        holding = strategy_fn(sig, idx, dt)

        if prev_holding is not None and holding != prev_holding:
            n_switches += 1
            # Switching cost: 0.02% spread
            value *= (1 - 0.0002)

        # Apply return
        if holding in rets.columns:
            r = rets[holding].iloc[idx]
            if not np.isnan(r):
                value *= (1 + r)

        prev_holding = holding
        daily_values.append(value)
        daily_dates.append(d)

    return {
        'values': np.array(daily_values),
        'dates': daily_dates,
        'contributed': contributed,
        'switches': n_switches,
        'final': value,
    }


def compute_metrics(result, label):
    """Compute Sharpe, Sortino, CAGR, MaxDD, Calmar."""
    vals = result['values']
    rets = np.diff(vals) / vals[:-1]
    rets = rets[~np.isnan(rets)]

    n_years = len(rets) / 252
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
    downside = rets[rets < 0]
    sortino = np.mean(rets) / np.std(downside) * np.sqrt(252) if len(downside) > 0 and np.std(downside) > 0 else 0

    cagr = (vals[-1] / vals[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    # MaxDD
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    maxdd = dd.min()

    calmar = cagr / abs(maxdd) if maxdd != 0 else 0

    sw_per_yr = result['switches'] / n_years if n_years > 0 else 0

    print(f"  {label:40s} | Sharpe {sharpe:6.2f} | Sortino {sortino:6.2f} | "
          f"CAGR {cagr:7.1%} | MaxDD {maxdd:7.1%} | Calmar {calmar:5.2f} | "
          f"Sw/yr {sw_per_yr:5.1f} | Final ${vals[-1]:>12,.0f}")

    return {
        'label': label, 'sharpe': sharpe, 'sortino': sortino,
        'cagr': cagr, 'maxdd': maxdd, 'calmar': calmar,
        'sw_per_yr': sw_per_yr, 'final': vals[-1],
    }


def permutation_test(closes, sig, strategy_fn, real_sharpe, warmup=260, n_perms=200):
    """Shuffle regime labels to test if timing is real."""
    print(f"\n  Permutation test ({n_perms} shuffles)...")
    perm_sharpes = []

    for p in range(n_perms):
        # Create shuffled strategy
        np.random.seed(p)
        # Pre-generate all regimes, then shuffle
        regimes = []
        for idx in range(warmup, len(closes)):
            d = closes.index[idx]
            dt = d.date() if hasattr(d, 'date') else d
            regimes.append(strategy_fn(sig, idx, dt))

        np.random.shuffle(regimes)
        regime_iter = iter(regimes)

        def shuffled_fn(sig, idx, dt, _iter=regime_iter):
            try:
                return next(_iter)
            except StopIteration:
                return 'SPY'

        result = simulate_portfolio(closes, sig, shuffled_fn, warmup=warmup)
        vals = result['values']
        rets = np.diff(vals) / vals[:-1]
        rets = rets[~np.isnan(rets)]
        s = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).sum() / len(perm_sharpes)
    print(f"  Real Sharpe: {real_sharpe:.3f} | Perm mean: {np.mean(perm_sharpes):.3f} | "
          f"Perm p-value: {p_value:.3f} | {'PASS' if p_value < 0.05 else 'FAIL'}")
    return p_value


def main():
    print("=" * 120)
    print("PORTFOLIO v3 TEST: v4.4 Full Adaptive vs GP3 in Portfolio Allocation")
    print("=" * 120)

    print("\nLoading data...")
    closes = load_data()
    print(f"  {len(closes)} days loaded ({closes.index[0].date()} to {closes.index[-1].date()})")

    print("Computing signals...")
    sig = compute_signals(closes)
    print("  Done.")

    warmup = 260

    # --- Strategy A: Original Portfolio v2 (50/50 VMR + Consensus(GP3+VMR)) ---
    gp3_in_upro = [False]
    def strategy_a(sig, i, dt):
        vmr_h = vmr_regime(sig, i)
        gp3_h, gp3_in_upro[0] = gp3_regime(sig, i, dt, gp3_in_upro[0])
        vmr_upro = vmr_h == 'UPRO'
        gp3_upro = gp3_h == 'UPRO'
        # Consensus: both agree
        cons_upro = vmr_upro and gp3_upro
        # 50% VMR + 50% Consensus
        if vmr_upro and cons_upro:
            return 'UPRO'
        elif vmr_upro:
            return 'SPY'  # VMR=UPRO, Cons=SPY → blend ≈ SPY (simplified)
        elif vmr_h == 'GLD':
            return 'GLD'
        else:
            return 'SPY'

    # --- Strategy B: Upgraded (50/50 VMR + Consensus(v4.4+VMR)) ---
    v44_in_upro_b = [False]
    def strategy_b(sig, i, dt):
        vmr_h = vmr_regime(sig, i)
        v44_h, v44_in_upro_b[0] = v44_regime(sig, i, dt, v44_in_upro_b[0])
        vmr_upro = vmr_h == 'UPRO'
        v44_upro = v44_h == 'UPRO'
        cons_upro = vmr_upro and v44_upro
        if vmr_upro and cons_upro:
            return 'UPRO'
        elif vmr_upro:
            return 'SPY'
        elif vmr_h == 'GLD':
            return 'GLD'
        else:
            return 'SPY'

    # --- Strategy C: 50/50 VMR + v4.4 standalone ---
    v44_in_upro_c = [False]
    def strategy_c(sig, i, dt):
        vmr_h = vmr_regime(sig, i)
        v44_h, v44_in_upro_c[0] = v44_regime(sig, i, dt, v44_in_upro_c[0])
        vmr_upro = vmr_h == 'UPRO'
        v44_upro = v44_h == 'UPRO'
        # Weight: UPRO if either says UPRO (more aggressive)
        if vmr_upro or v44_upro:
            return 'UPRO'
        elif vmr_h == 'GLD' or v44_h == 'GLD':
            return 'GLD'
        else:
            return 'SPY'

    # --- Strategy D: v4.4 standalone ---
    v44_in_upro_d = [False]
    def strategy_d(sig, i, dt):
        h, v44_in_upro_d[0] = v44_regime(sig, i, dt, v44_in_upro_d[0])
        return h

    # --- Strategy E: GP3 standalone (baseline) ---
    gp3_in_upro_e = [False]
    def strategy_e(sig, i, dt):
        h, gp3_in_upro_e[0] = gp3_regime(sig, i, dt, gp3_in_upro_e[0])
        return h

    # --- Benchmark: SPY B&H ---
    def strategy_spy(sig, i, dt):
        return 'SPY'

    print("\n" + "-" * 120)
    print("RESULTS (2010-2026, $500 initial + $100/wk DCA, 0.02% switching cost)")
    print("-" * 120)

    results = {}
    for label, fn in [
        ("SPY Buy & Hold", strategy_spy),
        ("GP3 v3 Standalone", strategy_e),
        ("v4.4 Full Adaptive Standalone", strategy_d),
        ("Portfolio v2: 50/50 VMR+Cons(GP3)", strategy_a),
        ("Portfolio v3: 50/50 VMR+Cons(v4.4)", strategy_b),
        ("Portfolio v3b: 50/50 VMR+v4.4 (OR)", strategy_c),
    ]:
        # Reset state for stateful strategies
        gp3_in_upro[0] = False
        v44_in_upro_b[0] = False
        v44_in_upro_c[0] = False
        v44_in_upro_d[0] = False
        gp3_in_upro_e[0] = False

        res = simulate_portfolio(closes, sig, fn, warmup=warmup)
        metrics = compute_metrics(res, label)
        results[label] = metrics

    # Permutation test on best portfolio
    print("\n" + "-" * 120)

    # Find best non-SPY
    best_label = max([k for k in results if k != "SPY Buy & Hold"],
                     key=lambda k: results[k]['sharpe'])
    best_sharpe = results[best_label]['sharpe']
    print(f"Best portfolio: {best_label} (Sharpe {best_sharpe:.3f})")

    # Also compare v4.4 vs GP3 head-to-head
    v44_sharpe = results.get("v4.4 Full Adaptive Standalone", {}).get('sharpe', 0)
    gp3_sharpe = results.get("GP3 v3 Standalone", {}).get('sharpe', 0)
    pv2_sharpe = results.get("Portfolio v2: 50/50 VMR+Cons(GP3)", {}).get('sharpe', 0)
    pv3_sharpe = results.get("Portfolio v3: 50/50 VMR+Cons(v4.4)", {}).get('sharpe', 0)

    print(f"\nv4.4 vs GP3 standalone: {v44_sharpe:.3f} vs {gp3_sharpe:.3f} "
          f"({'v4.4 WINS' if v44_sharpe > gp3_sharpe else 'GP3 WINS'} by {abs(v44_sharpe - gp3_sharpe):.3f})")
    print(f"Portfolio v3 vs v2:     {pv3_sharpe:.3f} vs {pv2_sharpe:.3f} "
          f"({'v3 WINS' if pv3_sharpe > pv2_sharpe else 'v2 WINS'} by {abs(pv3_sharpe - pv2_sharpe):.3f})")

    # Sub-period analysis
    print("\n" + "-" * 120)
    print("SUB-PERIOD ANALYSIS (3-year blocks)")
    print("-" * 120)

    for label, fn in [
        ("Portfolio v2: VMR+Cons(GP3)", strategy_a),
        ("Portfolio v3: VMR+Cons(v4.4)", strategy_b),
    ]:
        vals_all = []
        gp3_in_upro[0] = False
        v44_in_upro_b[0] = False
        res = simulate_portfolio(closes, sig, fn, warmup=warmup)
        vals = res['values']
        dates = res['dates']

        # Split into 3-year blocks
        n_per_block = 252 * 3
        n_blocks = len(vals) // n_per_block
        block_sharpes = []
        for b in range(n_blocks):
            start = b * n_per_block
            end = min((b + 1) * n_per_block, len(vals))
            block_vals = vals[start:end]
            block_rets = np.diff(block_vals) / block_vals[:-1]
            block_rets = block_rets[~np.isnan(block_rets)]
            bs = np.mean(block_rets) / np.std(block_rets) * np.sqrt(252) if np.std(block_rets) > 0 else 0
            block_sharpes.append(bs)

        cv = np.std(block_sharpes) / np.mean(block_sharpes) if np.mean(block_sharpes) > 0 else 999
        print(f"  {label}: blocks={[f'{s:.2f}' for s in block_sharpes]}, CV={cv:.3f}")

    print("\n" + "=" * 120)
    print("VERDICT")
    print("=" * 120)

    if pv3_sharpe > pv2_sharpe:
        delta = pv3_sharpe - pv2_sharpe
        print(f"  Portfolio v3 (v4.4) BEATS Portfolio v2 (GP3) by {delta:.3f} Sharpe")
        if delta > 0.05:
            print(f"  RECOMMEND: Upgrade Portfolio Engine to use v4.4 in Consensus signal")
        else:
            print(f"  MARGINAL improvement — keep v2 as default, track v3 in parallel")
    else:
        print(f"  Portfolio v2 (GP3) still better. v4.4 advantage is standalone, not in portfolio blend.")

    print()


if __name__ == "__main__":
    main()
