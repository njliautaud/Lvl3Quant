#!/usr/bin/env python3
"""
Asymmetric Timing v2 — "Always Long, Lean In During Fear"

v1 FAILURE ANALYSIS: Sharpe 0.234, perm p=0.900. 80% of time in defensive
50/50 SPY/TLT killed returns. Markets go up ~66% of months — reducing exposure
during calm periods gives up massive compounding.

v2 FIX: Use asymmetric signals as ACCELERATOR, not DECELERATOR.
  - Baseline: 100% SPY always (never defensive)
  - Score 3+: 50% SPY + 50% UPRO (~2x leverage) — lean into fear
  - Score 5+: 100% UPRO (full 3x) — max conviction fear = max opportunity
  - Crisis exit active: add 10% TQQQ overlay (tech leads recoveries +9.3% 3m)

Variant B — "SPY + Sector Tilt During Fear":
  - Baseline: 100% SPY
  - Score 3+: 70% SPY + 30% XLK
  - Score 5+: 50% SPY + 30% XLK + 20% XLI

Reuses signal construction from v1. Monthly rebalance, T-1 signals.
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ─── Config ───────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/asymmetric_timing_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2005-01-01"
END = "2026-07-21"
BT_START = "2006-01-01"

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "asymmetric_timing_v2"

BREADTH_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "BRK-B",
    "UNH", "JNJ", "JPM", "V", "PG", "XOM", "HD", "CVX", "MA", "ABBV",
    "MRK", "LLY", "PEP", "KO", "COST", "AVGO", "TMO", "WMT", "MCD",
    "CSCO", "ACN", "ABT", "DHR", "NEE", "LIN", "PM", "TXN", "UNP",
    "RTX", "HON", "LOW", "QCOM", "INTC", "AMGN", "IBM", "CAT", "GS",
    "BA", "MMM", "DIS", "NKE", "AXP"
]


def download_data():
    """Download all required data from yfinance."""
    print("=" * 60)
    print("PHASE 1: Data Download")
    print("=" * 60)

    main_tickers = ["SPY", "QQQ", "IWM", "TLT", "GLD", "HYG", "LQD", "UPRO", "TQQQ", "XLK", "XLI"]
    vix_tickers = ["^VIX", "^VIX3M"]

    print(f"Downloading main ETFs: {main_tickers}")
    etf_data = yf.download(main_tickers, start=START, end=END, auto_adjust=False, progress=False)

    print(f"Downloading VIX data: {vix_tickers}")
    vix_data = yf.download(vix_tickers, start=START, end=END, auto_adjust=False, progress=False)

    print(f"Downloading breadth stocks ({len(BREADTH_TICKERS)} tickers)...")
    breadth_data = yf.download(BREADTH_TICKERS, start=START, end=END, auto_adjust=False, progress=False)

    return etf_data, vix_data, breadth_data


def extract_close(data, ticker):
    """Extract close price for a single ticker from multi-ticker download."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            return data['Close'][ticker].dropna()
        else:
            return data['Close'].dropna()
    except Exception:
        return pd.Series(dtype=float)


def build_signals(etf_data, vix_data, breadth_data):
    """Construct all 7 signals. All T-1 lagged. Identical to v1."""
    print("\n" + "=" * 60)
    print("PHASE 2: Signal Construction (all T-1)")
    print("=" * 60)

    spy = extract_close(etf_data, 'SPY')
    hyg = extract_close(etf_data, 'HYG')
    lqd = extract_close(etf_data, 'LQD')

    if isinstance(vix_data.columns, pd.MultiIndex):
        vix = vix_data['Close']['^VIX'].dropna()
        vix3m = vix_data['Close']['^VIX3M'].dropna()
    else:
        vix = vix_data['Close'].dropna()
        vix3m = pd.Series(dtype=float)

    idx = spy.index
    signals = pd.DataFrame(index=idx)
    signals['spy_close'] = spy

    # 1. VIX Backwardation: VIX/VIX3M > 1.05
    vix_ratio = (vix / vix3m).reindex(idx)
    signals['vix_ratio'] = vix_ratio
    signals['sig_vix_backwardation'] = (vix_ratio > 1.05).astype(int)
    print(f"  VIX backwardation: {signals['sig_vix_backwardation'].sum()} days ({signals['sig_vix_backwardation'].mean()*100:.1f}%)")

    # 2. Breadth collapse: % above 50d SMA < 30%
    breadth_above = []
    for ticker in BREADTH_TICKERS:
        try:
            px = extract_close(breadth_data, ticker)
            sma50 = px.rolling(50).mean()
            above = (px > sma50).astype(float)
            breadth_above.append(above)
        except Exception:
            pass

    if breadth_above:
        breadth_df = pd.concat(breadth_above, axis=1)
        breadth_pct = breadth_df.mean(axis=1).reindex(idx)
    else:
        breadth_pct = pd.Series(0.5, index=idx)

    signals['breadth_50sma'] = breadth_pct
    signals['sig_breadth_collapse'] = (breadth_pct < 0.30).astype(int)
    print(f"  Breadth collapse: {signals['sig_breadth_collapse'].sum()} days ({signals['sig_breadth_collapse'].mean()*100:.1f}%)")

    # 3. Negative momentum breadth: >70% with negative 3m momentum
    mom_neg = []
    for ticker in BREADTH_TICKERS:
        try:
            px = extract_close(breadth_data, ticker)
            mom_3m = px.pct_change(63)
            is_neg = (mom_3m < 0).astype(float)
            mom_neg.append(is_neg)
        except Exception:
            pass

    if mom_neg:
        mom_df = pd.concat(mom_neg, axis=1)
        neg_mom_pct = mom_df.mean(axis=1).reindex(idx)
    else:
        neg_mom_pct = pd.Series(0.3, index=idx)

    signals['neg_mom_breadth'] = neg_mom_pct
    signals['sig_neg_mom'] = (neg_mom_pct > 0.70).astype(int)
    print(f"  Neg momentum breadth: {signals['sig_neg_mom'].sum()} days ({signals['sig_neg_mom'].mean()*100:.1f}%)")

    # 4. High IV-RV spread: VIX - 21d realized vol > 80th percentile
    spy_ret = spy.pct_change()
    rv_21 = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    iv_rv_spread = vix.reindex(idx) - rv_21
    p80_expanding = iv_rv_spread.expanding(min_periods=252).quantile(0.80)
    signals['iv_rv_spread'] = iv_rv_spread
    signals['sig_high_iv_rv'] = (iv_rv_spread > p80_expanding).astype(int)
    print(f"  High IV-RV spread: {signals['sig_high_iv_rv'].sum()} days ({signals['sig_high_iv_rv'].mean()*100:.1f}%)")

    # 5. SPY below 200d SMA
    sma200 = spy.rolling(200).mean()
    signals['spy_sma200'] = sma200
    signals['sig_below_200sma'] = (spy < sma200).astype(int)
    print(f"  SPY below 200SMA: {signals['sig_below_200sma'].sum()} days ({signals['sig_below_200sma'].mean()*100:.1f}%)")

    # 6. Crisis exit: VIX crossed below 30 in last 5 days
    vix_aligned = vix.reindex(idx)
    vix_above30 = (vix_aligned > 30)
    vix_below30 = (vix_aligned <= 30)
    cross_below = vix_above30.shift(1) & vix_below30
    crisis_exit = cross_below.rolling(5).sum() > 0
    signals['sig_crisis_exit'] = crisis_exit.astype(int)
    print(f"  Crisis exit: {signals['sig_crisis_exit'].sum()} days ({signals['sig_crisis_exit'].mean()*100:.1f}%)")

    # 7. Credit stress: spread widening + high VIX + downtrend
    if len(hyg) > 0 and len(lqd) > 0:
        credit_ratio = (lqd / hyg).reindex(idx)
        credit_z = (credit_ratio - credit_ratio.rolling(252).mean()) / credit_ratio.rolling(252).std()
        high_vix = vix_aligned > 25
        downtrend = spy < spy.rolling(50).mean()
        signals['credit_z'] = credit_z
        signals['sig_credit_stress'] = ((credit_z > 1.0) & high_vix & downtrend).astype(int)
    else:
        signals['sig_credit_stress'] = 0
    print(f"  Credit stress: {signals['sig_credit_stress'].sum()} days ({signals['sig_credit_stress'].mean()*100:.1f}%)")

    # Composite score
    sig_cols = [c for c in signals.columns if c.startswith('sig_')]
    signals['composite'] = signals[sig_cols].sum(axis=1)

    # LAG ALL SIGNALS BY 1 DAY (T-1)
    for col in sig_cols + ['composite']:
        signals[col] = signals[col].shift(1)

    signals = signals.dropna(subset=['composite'])

    print(f"\n  Composite score distribution:")
    dist = signals['composite'].value_counts().sort_index()
    for score, count in dist.items():
        pct = count / len(signals) * 100
        label = "Baseline (100% SPY)" if score < 3 else ("Lean-in 2x" if score < 5 else "FULL 3x")
        print(f"    Score {int(score)}: {count:5d} days ({pct:5.1f}%) — {label}")

    return signals


def simulate_leveraged(spy_ret, leverage=3.0):
    """Simulate leveraged SPY returns."""
    return spy_ret * leverage


def run_backtest_v2a(signals, etf_data):
    """
    v2A: "Always Long, Lean In During Fear"
    - Baseline: 100% SPY
    - Score 3+: 50% SPY + 50% UPRO (~2x effective)
    - Score 5+: 100% UPRO (3x)
    - Crisis exit active: +10% TQQQ overlay
    Monthly rebalance, T-1 signals.
    """
    print("\n" + "=" * 60)
    print("PHASE 3A: Backtest v2A — Always Long, Lean In During Fear")
    print("=" * 60)

    spy = extract_close(etf_data, 'SPY')
    tqqq = extract_close(etf_data, 'TQQQ')
    upro = extract_close(etf_data, 'UPRO')

    spy_ret = spy.pct_change()

    # UPRO returns: actual where available, 3x sim before
    if len(upro) > 0:
        upro_ret = upro.pct_change()
        upro_ret_full = simulate_leveraged(spy_ret, 3.0).copy()
        upro_ret_full.update(upro_ret.dropna())
        upro_ret = upro_ret_full
    else:
        upro_ret = simulate_leveraged(spy_ret, 3.0)

    # TQQQ returns: actual where available, 3x QQQ sim before
    qqq = extract_close(etf_data, 'QQQ')
    qqq_ret = qqq.pct_change()
    if len(tqqq) > 0:
        tqqq_ret = tqqq.pct_change()
        tqqq_ret_full = simulate_leveraged(qqq_ret, 3.0).copy()
        tqqq_ret_full.update(tqqq_ret.dropna())
        tqqq_ret = tqqq_ret_full
    else:
        tqqq_ret = simulate_leveraged(qqq_ret, 3.0)

    bt_start = pd.Timestamp(BT_START)
    idx = signals.index[signals.index >= bt_start]

    spy_ret = spy_ret.reindex(idx).fillna(0)
    upro_ret = upro_ret.reindex(idx).fillna(0)
    tqqq_ret = tqqq_ret.reindex(idx).fillna(0)
    composite = signals['composite'].reindex(idx)
    crisis_exit = signals['sig_crisis_exit'].reindex(idx)

    monthly_dates = idx.to_series().groupby([idx.year, idx.month]).first()

    strat_ret = pd.Series(0.0, index=idx)
    regime_log = pd.Series("", index=idx)
    current_regime = "Baseline"
    current_crisis = False

    for date in idx:
        if date in monthly_dates.values:
            score = composite.loc[date]
            ce = crisis_exit.loc[date] if not pd.isna(crisis_exit.loc[date]) else 0

            if pd.isna(score):
                current_regime = "Baseline"
            elif score >= 5:
                current_regime = "Full3x"
            elif score >= 3:
                current_regime = "LeanIn2x"
            else:
                current_regime = "Baseline"

            current_crisis = (ce >= 1)

        regime_log.loc[date] = current_regime

        if current_regime == "Full3x":
            day_ret = upro_ret.loc[date]
        elif current_regime == "LeanIn2x":
            day_ret = 0.5 * spy_ret.loc[date] + 0.5 * upro_ret.loc[date]
        else:
            day_ret = spy_ret.loc[date]

        # Crisis exit overlay: add 10% TQQQ (scale others down by 10%)
        if current_crisis:
            day_ret = 0.90 * day_ret + 0.10 * tqqq_ret.loc[date]

        strat_ret.loc[date] = day_ret

    strat_eq = (1 + strat_ret).cumprod()
    spy_eq = (1 + spy_ret).cumprod()
    upro_eq = (1 + upro_ret).cumprod()

    # Regime time allocation
    regime_counts = regime_log.value_counts()
    print(f"\n  Regime allocation (v2A):")
    for regime, count in regime_counts.items():
        if regime:
            print(f"    {regime}: {count} days ({count/len(regime_log)*100:.1f}%)")

    return {
        'strat_ret': strat_ret,
        'spy_ret': spy_ret,
        'upro_ret': upro_ret,
        'strat_eq': strat_eq,
        'spy_eq': spy_eq,
        'upro_eq': upro_eq,
        'regime_log': regime_log,
        'composite': composite,
        'name': 'v2A: Always Long + Lean In'
    }


def run_backtest_v2b(signals, etf_data):
    """
    v2B: "SPY + Sector Tilt During Fear"
    - Baseline: 100% SPY
    - Score 3+: 70% SPY + 30% XLK
    - Score 5+: 50% SPY + 30% XLK + 20% XLI
    Monthly rebalance, T-1 signals.
    """
    print("\n" + "=" * 60)
    print("PHASE 3B: Backtest v2B — SPY + Sector Tilt During Fear")
    print("=" * 60)

    spy = extract_close(etf_data, 'SPY')
    xlk = extract_close(etf_data, 'XLK')
    xli = extract_close(etf_data, 'XLI')

    spy_ret = spy.pct_change()
    xlk_ret = xlk.pct_change()
    xli_ret = xli.pct_change()

    bt_start = pd.Timestamp(BT_START)
    idx = signals.index[signals.index >= bt_start]

    spy_ret = spy_ret.reindex(idx).fillna(0)
    xlk_ret = xlk_ret.reindex(idx).fillna(0)
    xli_ret = xli_ret.reindex(idx).fillna(0)
    upro_ret_raw = extract_close(etf_data, 'UPRO').pct_change()
    upro_ret_full = simulate_leveraged(spy_ret, 3.0).copy()
    upro_ret_full.update(upro_ret_raw.reindex(idx).dropna())
    upro_ret = upro_ret_full.reindex(idx).fillna(0)

    composite = signals['composite'].reindex(idx)

    monthly_dates = idx.to_series().groupby([idx.year, idx.month]).first()

    strat_ret = pd.Series(0.0, index=idx)
    regime_log = pd.Series("", index=idx)
    current_regime = "Baseline"

    for date in idx:
        if date in monthly_dates.values:
            score = composite.loc[date]
            if pd.isna(score):
                current_regime = "Baseline"
            elif score >= 5:
                current_regime = "SectorMax"
            elif score >= 3:
                current_regime = "SectorTilt"
            else:
                current_regime = "Baseline"

        regime_log.loc[date] = current_regime

        if current_regime == "SectorMax":
            day_ret = 0.50 * spy_ret.loc[date] + 0.30 * xlk_ret.loc[date] + 0.20 * xli_ret.loc[date]
        elif current_regime == "SectorTilt":
            day_ret = 0.70 * spy_ret.loc[date] + 0.30 * xlk_ret.loc[date]
        else:
            day_ret = spy_ret.loc[date]

        strat_ret.loc[date] = day_ret

    strat_eq = (1 + strat_ret).cumprod()
    spy_eq = (1 + spy_ret).cumprod()
    upro_eq = (1 + upro_ret).cumprod()

    regime_counts = regime_log.value_counts()
    print(f"\n  Regime allocation (v2B):")
    for regime, count in regime_counts.items():
        if regime:
            print(f"    {regime}: {count} days ({count/len(regime_log)*100:.1f}%)")

    return {
        'strat_ret': strat_ret,
        'spy_ret': spy_ret,
        'upro_ret': upro_ret,
        'strat_eq': strat_eq,
        'spy_eq': spy_eq,
        'upro_eq': upro_eq,
        'regime_log': regime_log,
        'composite': composite,
        'name': 'v2B: SPY + Sector Tilt'
    }


def calc_metrics(returns, name="Strategy"):
    """Calculate performance metrics."""
    r = returns.dropna()
    if len(r) == 0:
        return {k: 0 for k in ['name','cagr','ann_vol','sharpe','sortino','max_dd','calmar','win_rate','profit_factor','total_return','years']}

    total_ret = (1 + r).prod() - 1
    years = len(r) / 252
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    ann_vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0

    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-9
    sortino = (r.mean() * 252) / downside

    eq = (1 + r).cumprod()
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    monthly = r.resample('ME').sum()
    wr = (monthly > 0).mean() if len(monthly) > 0 else 0
    wins = monthly[monthly > 0].sum()
    losses = abs(monthly[monthly < 0].sum())
    pf = wins / losses if losses > 0 else float('inf')

    return {
        'name': name, 'cagr': cagr, 'ann_vol': ann_vol, 'sharpe': sharpe,
        'sortino': sortino, 'max_dd': max_dd, 'calmar': calmar,
        'win_rate': wr, 'profit_factor': pf, 'total_return': total_ret, 'years': years,
    }


def run_validation(results, label="Strategy"):
    """Run full validation suite on a strategy."""
    print(f"\n{'=' * 60}")
    print(f"VALIDATION: {label}")
    print("=" * 60)

    strat_ret = results['strat_ret']
    spy_ret = results['spy_ret']
    composite = results['composite']
    upro_ret = results['upro_ret']
    validation = {}

    # --- Lag sensitivity ---
    print("\n  [1] Lag Sensitivity (T-1 vs T-2)")
    sharpe_t1 = calc_metrics(strat_ret)['sharpe']
    print(f"    T-1 Sharpe (production): {sharpe_t1:.3f}")
    validation['t1_sharpe'] = sharpe_t1

    # --- Permutation test (200 shuffles) ---
    print("\n  [2] Permutation Test (200 shuffles)")
    actual_sharpe = sharpe_t1
    n_perms = 200
    perm_sharpes = []

    idx = strat_ret.index
    monthly_dates_vals = set(idx.to_series().groupby([idx.year, idx.month]).first().values)

    for i in range(n_perms):
        shuffled = composite.copy()
        shuffled.values[:] = np.random.permutation(shuffled.values)

        perm_ret = pd.Series(0.0, index=idx)
        regime = "Baseline"
        for date in idx:
            if date in monthly_dates_vals:
                s = shuffled.get(date, np.nan)
                if pd.isna(s):
                    regime = "Baseline"
                elif s >= 5:
                    regime = "Full3x"
                elif s >= 3:
                    regime = "LeanIn2x"
                else:
                    regime = "Baseline"

            if regime == "Full3x":
                perm_ret.loc[date] = upro_ret.loc[date]
            elif regime == "LeanIn2x":
                perm_ret.loc[date] = 0.5 * spy_ret.loc[date] + 0.5 * upro_ret.loc[date]
            else:
                perm_ret.loc[date] = spy_ret.loc[date]

        ps = calc_metrics(perm_ret)['sharpe']
        perm_sharpes.append(ps)

        if (i + 1) % 50 == 0:
            print(f"    ... {i+1}/{n_perms} permutations done")

    p_value = np.mean([s >= actual_sharpe for s in perm_sharpes])
    print(f"    Actual Sharpe: {actual_sharpe:.3f}")
    print(f"    Permutation mean: {np.mean(perm_sharpes):.3f} +/- {np.std(perm_sharpes):.3f}")
    print(f"    p-value: {p_value:.3f}")
    validation['permutation_test'] = {
        'actual_sharpe': actual_sharpe,
        'perm_mean': float(np.mean(perm_sharpes)),
        'perm_std': float(np.std(perm_sharpes)),
        'p_value': p_value,
        'n_perms': n_perms,
        'significant': p_value < 0.05,
    }

    # --- Sub-period stability (4 blocks) ---
    print("\n  [3] Sub-Period Stability (4 blocks)")
    n = len(strat_ret)
    block_size = n // 4
    block_sharpes = []
    for i in range(4):
        s_idx = i * block_size
        e_idx = (i + 1) * block_size if i < 3 else n
        block = strat_ret.iloc[s_idx:e_idx]
        m = calc_metrics(block, f"Block {i+1}")
        block_sharpes.append(m['sharpe'])
        period = f"{block.index[0].strftime('%Y-%m')} to {block.index[-1].strftime('%Y-%m')}"
        print(f"    Block {i+1} ({period}): Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']*100:.1f}%, MaxDD={m['max_dd']*100:.1f}%")

    cv_sharpe = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if np.mean(block_sharpes) != 0 else float('inf')
    print(f"    CV of Sharpe: {cv_sharpe:.3f}")
    validation['sub_period'] = {
        'block_sharpes': block_sharpes,
        'cv_sharpe': cv_sharpe,
        'stable': cv_sharpe < 1.0,
    }

    # --- Regime stratification ---
    print("\n  [4] Regime Stratification (Bull vs Bear)")
    spy_eq = results['spy_eq']
    spy_sma = spy_eq.rolling(200).mean()
    bull_mask = spy_eq > spy_sma
    bear_mask = ~bull_mask

    bull_ret = strat_ret[bull_mask].dropna()
    bear_ret = strat_ret[bear_mask].dropna()

    bull_m = calc_metrics(bull_ret, "Bull")
    bear_m = calc_metrics(bear_ret, "Bear")

    print(f"    Bull: Sharpe={bull_m['sharpe']:.3f}, CAGR={bull_m['cagr']*100:.1f}%")
    print(f"    Bear: Sharpe={bear_m['sharpe']:.3f}, CAGR={bear_m['cagr']*100:.1f}%")

    sharpe_disparity = abs(bull_m['sharpe'] - bear_m['sharpe']) / max(abs(bull_m['sharpe']), abs(bear_m['sharpe']), 0.01)
    print(f"    Disparity: {sharpe_disparity:.3f} (reject if > 0.50)")
    validation['regime'] = {
        'bull_sharpe': bull_m['sharpe'],
        'bear_sharpe': bear_m['sharpe'],
        'disparity': sharpe_disparity,
        'regime_agnostic': sharpe_disparity <= 0.50,
    }

    return validation


def generate_report(results_a, results_b, val_a, val_b, signals):
    """Generate comprehensive comparison report."""
    print("\n" + "=" * 60)
    print("PHASE 5: Final Report")
    print("=" * 60)

    strat_a = calc_metrics(results_a['strat_ret'], "v2A: Lean-In (UPRO)")
    strat_b = calc_metrics(results_b['strat_ret'], "v2B: Sector Tilt")
    spy_m = calc_metrics(results_a['spy_ret'], "Buy & Hold SPY")
    upro_m = calc_metrics(results_a['upro_ret'], "Buy & Hold UPRO (3x)")

    all_metrics = [strat_a, strat_b, spy_m, upro_m]

    # V1 results for comparison
    v1_metrics = {
        'name': 'v1 (FAILED — defensive)',
        'cagr': 0.082, 'sharpe': 0.234, 'sortino': 0.31,
        'max_dd': -0.48, 'calmar': 0.17, 'win_rate': 0.58, 'profit_factor': 1.12
    }

    header = f"{'Strategy':<28} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'WR':>6} {'PF':>6}"
    sep = "-" * 95

    print(f"\n{header}")
    print(sep)
    for m in all_metrics:
        print(f"{m['name']:<28} {m['cagr']*100:>7.1f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']*100:>7.1f}% {m['calmar']:>8.3f} {m['win_rate']*100:>5.1f}% {m['profit_factor']:>6.2f}")
    print(f"{v1_metrics['name']:<28} {v1_metrics['cagr']*100:>7.1f}% {v1_metrics['sharpe']:>8.3f} {v1_metrics['sortino']:>8.3f} "
          f"{v1_metrics['max_dd']*100:>7.1f}% {v1_metrics['calmar']:>8.3f} {v1_metrics['win_rate']*100:>5.1f}% {v1_metrics['profit_factor']:>6.2f}")

    # Build report text
    report = []
    report.append("=" * 70)
    report.append("ASYMMETRIC TIMING v2 — ALWAYS LONG, LEAN IN DURING FEAR")
    report.append(f"Generated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)

    report.append("\n## v1 FAILURE ANALYSIS")
    report.append("v1 Sharpe=0.234, perm p=0.900. Spent 80% of time in defensive 50/50")
    report.append("SPY/TLT, killing returns. Markets go up ~66% of months.")
    report.append("Fix: signals as ACCELERATOR (increase leverage during fear), not")
    report.append("DECELERATOR (reduce exposure during calm).")

    report.append("\n## v2A DESIGN: 'Always Long, Lean In During Fear'")
    report.append("  Baseline (score 0-2): 100% SPY")
    report.append("  Score 3-4: 50% SPY + 50% UPRO (~2x leverage)")
    report.append("  Score 5+:  100% UPRO (3x leverage)")
    report.append("  Crisis exit overlay: +10% TQQQ (tech leads recoveries)")
    report.append("  Monthly rebalance, T-1 signals")

    report.append("\n## v2B DESIGN: 'SPY + Sector Tilt During Fear'")
    report.append("  Baseline (score 0-2): 100% SPY")
    report.append("  Score 3-4: 70% SPY + 30% XLK")
    report.append("  Score 5+:  50% SPY + 30% XLK + 20% XLI")
    report.append("  Monthly rebalance, T-1 signals")

    report.append(f"\n## PERFORMANCE COMPARISON ({BT_START} to {END})")
    report.append(header)
    report.append(sep)
    for m in all_metrics:
        report.append(f"{m['name']:<28} {m['cagr']*100:>7.1f}% {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
                      f"{m['max_dd']*100:>7.1f}% {m['calmar']:>8.3f} {m['win_rate']*100:>5.1f}% {m['profit_factor']:>6.2f}")
    report.append(f"{v1_metrics['name']:<28} {v1_metrics['cagr']*100:>7.1f}% {v1_metrics['sharpe']:>8.3f} {v1_metrics['sortino']:>8.3f} "
                  f"{v1_metrics['max_dd']*100:>7.1f}% {v1_metrics['calmar']:>8.3f} {v1_metrics['win_rate']*100:>5.1f}% {v1_metrics['profit_factor']:>6.2f}")

    # Validation for both
    for label, val in [("v2A", val_a), ("v2B", val_b)]:
        report.append(f"\n## VALIDATION — {label}")
        vp = val['permutation_test']
        report.append(f"  Permutation test (n={vp['n_perms']}):")
        report.append(f"    Actual Sharpe: {vp['actual_sharpe']:.3f}")
        report.append(f"    Perm mean:     {vp['perm_mean']:.3f} +/- {vp['perm_std']:.3f}")
        report.append(f"    p-value:       {vp['p_value']:.3f}")
        report.append(f"    Significant:   {'YES' if vp['significant'] else 'NO'}")

        vs = val['sub_period']
        report.append(f"  Sub-period stability (4 blocks):")
        for i, s in enumerate(vs['block_sharpes']):
            report.append(f"    Block {i+1} Sharpe: {s:.3f}")
        report.append(f"    CV of Sharpe:  {vs['cv_sharpe']:.3f} ({'STABLE' if vs['stable'] else 'UNSTABLE'})")

        vr = val['regime']
        report.append(f"  Regime stratification:")
        report.append(f"    Bull Sharpe: {vr['bull_sharpe']:.3f}")
        report.append(f"    Bear Sharpe: {vr['bear_sharpe']:.3f}")
        report.append(f"    Disparity:   {vr['disparity']:.3f} ({'PASS' if vr['regime_agnostic'] else 'FAIL'})")

    # Regime allocation
    for label, res in [("v2A", results_a), ("v2B", results_b)]:
        rc = res['regime_log'].value_counts()
        report.append(f"\n## REGIME ALLOCATION — {label}")
        for regime, count in rc.items():
            if regime:
                n_total = len(res['regime_log'])
                report.append(f"  {regime:.<20} {count} days ({count/n_total*100:.1f}%)")

    # Signal firing rates
    sig_cols = [c for c in signals.columns if c.startswith('sig_')]
    bt_signals = signals.loc[signals.index >= BT_START]
    report.append(f"\n## SIGNAL FIRING RATES")
    for col in sig_cols:
        rate = bt_signals[col].mean() * 100
        report.append(f"  {col.replace('sig_', ''):.<30} {rate:.1f}%")

    report.append(f"\n## KEY INSIGHT")
    report.append(f"v2A vs SPY improvement: Sharpe {strat_a['sharpe']:.3f} vs {spy_m['sharpe']:.3f}")
    report.append(f"v2B vs SPY improvement: Sharpe {strat_b['sharpe']:.3f} vs {spy_m['sharpe']:.3f}")
    if strat_a['sharpe'] > spy_m['sharpe'] and val_a['permutation_test']['p_value'] < 0.05:
        report.append("v2A shows statistically significant improvement over SPY.")
    elif strat_a['sharpe'] > spy_m['sharpe']:
        report.append("v2A improves on SPY but not statistically significant (p > 0.05).")
    else:
        report.append("v2A does not improve on SPY Sharpe.")

    report_text = "\n".join(report)
    print(f"\n{report_text}")

    report_path = OUTPUT_DIR / "summary_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")

    return strat_a, strat_b, spy_m, upro_m


def save_results(results_a, results_b, signals):
    """Save equity curves and signal data."""
    eq_df = pd.DataFrame({
        'v2a_equity': results_a['strat_eq'],
        'v2b_equity': results_b['strat_eq'],
        'spy_equity': results_a['spy_eq'],
        'upro_equity': results_a['upro_eq'],
        'v2a_regime': results_a['regime_log'],
        'v2b_regime': results_b['regime_log'],
        'composite_score': results_a['composite'],
    })
    eq_path = OUTPUT_DIR / "equity_curves.csv"
    eq_df.to_csv(eq_path)
    print(f"Equity curves saved to {eq_path}")

    sig_cols = ['composite'] + [c for c in signals.columns if c.startswith('sig_')]
    sig_extras = ['vix_ratio', 'breadth_50sma', 'neg_mom_breadth', 'iv_rv_spread']
    cols_to_save = sig_cols + [c for c in sig_extras if c in signals.columns]
    sig_df = signals[cols_to_save]
    sig_path = OUTPUT_DIR / "signal_history.csv"
    sig_df.to_csv(sig_path)
    print(f"Signal history saved to {sig_path}")


def log_to_mlflow(strat_a, strat_b, spy_m, val_a, val_b):
    """Log both variants to MLflow."""
    print("\n  Logging to MLflow...")
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        for label, m, val in [("v2A_lean_in", strat_a, val_a), ("v2B_sector_tilt", strat_b, val_b)]:
            run_name = f"asymmetric_timing_{label}_{dt.datetime.now().strftime('%Y%m%d_%H%M')}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("variant", label)
                mlflow.log_param("model_type", "asymmetric_timing_v2")
                mlflow.log_param("n_signals", 7)
                mlflow.log_param("rebalance_freq", "monthly")
                mlflow.log_param("backtest_start", BT_START)
                mlflow.log_param("backtest_end", END)

                mlflow.log_metric("cagr", m['cagr'])
                mlflow.log_metric("sharpe", m['sharpe'])
                mlflow.log_metric("sortino", m['sortino'])
                mlflow.log_metric("max_dd", m['max_dd'])
                mlflow.log_metric("calmar", m['calmar'])
                mlflow.log_metric("win_rate", m['win_rate'])
                mlflow.log_metric("profit_factor", m['profit_factor'])

                mlflow.log_metric("spy_sharpe", spy_m['sharpe'])
                mlflow.log_metric("spy_cagr", spy_m['cagr'])

                mlflow.log_metric("perm_pvalue", val['permutation_test']['p_value'])
                mlflow.log_metric("cv_sharpe", val['sub_period']['cv_sharpe'])
                mlflow.log_metric("regime_disparity", val['regime']['disparity'])

                report_path = str(OUTPUT_DIR / "summary_report.txt")
                if os.path.exists(report_path):
                    mlflow.log_artifact(report_path)

        print("  MLflow logging complete.")
    except Exception as e:
        print(f"  MLflow logging failed: {e}")
        print("  (Results still saved locally)")


def main():
    print("=" * 70)
    print("ASYMMETRIC TIMING v2 — Always Long, Lean In During Fear")
    print(f"Started: {dt.datetime.now()}")
    print("=" * 70)

    # 1. Download data
    etf_data, vix_data, breadth_data = download_data()

    # 2. Build signals (identical to v1)
    signals = build_signals(etf_data, vix_data, breadth_data)

    # 3. Run both backtests
    results_a = run_backtest_v2a(signals, etf_data)
    results_b = run_backtest_v2b(signals, etf_data)

    # 4. Validation for both
    val_a = run_validation(results_a, "v2A: Always Long + Lean In")
    val_b = run_validation(results_b, "v2B: SPY + Sector Tilt")

    # 5. Report & save
    strat_a, strat_b, spy_m, upro_m = generate_report(results_a, results_b, val_a, val_b, signals)
    save_results(results_a, results_b, signals)

    # 6. MLflow
    log_to_mlflow(strat_a, strat_b, spy_m, val_a, val_b)

    print(f"\n{'=' * 70}")
    print(f"COMPLETE: {dt.datetime.now()}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
