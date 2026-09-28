#!/usr/bin/env python3
"""
R5 TASK 1: MAXIMIZE LEVERAGED RISK PARITY
HC #697: No crypto | HC #0: Sliding window only | HC #694: Commission-free
HC #428: Regime test (report honestly, user accepts more risk for growth)

Sub-tasks:
A) Leverage sweep: 1.5x, 2x, 2.5x, 3x, 3.5x — CAGR vs MaxDD vs regime gap frontier
B) Asset universe expansion: add EFA, EEM, VNQ, TIP, DBMF
C) Smarter predictive tilt: carry, momentum 12-1, mean reversion RSI, vol targeting
D) Implementation: margin vs leveraged ETFs (UPRO/TMF/UGL vol drag)

Walk-forward: 252d train, 21d test, sliding window.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
import json
import os
from datetime import datetime

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r5'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2010-01-01'
END = '2026-07-14'


# ─── Shared Utilities ───────────────────────────────────────────────────────

def calc_metrics(returns, name=''):
    rets = returns.dropna()
    if len(rets) < 20:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0,
                'max_dd': 0, 'win_rate': 0, 'pf': 0, 'n_days': len(rets)}
    total_ret = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = rets.mean() * 252 / downside if downside > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    if max_dd < -1.0:
        max_dd = -1.0
    wr = (rets > 0).mean()
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    return {'name': name, 'sharpe': float(sharpe), 'sortino': float(sortino),
            'cagr': float(cagr), 'max_dd': float(max_dd), 'win_rate': float(wr),
            'pf': float(pf), 'n_days': int(len(rets)),
            'total_return_pct': float(total_ret * 100),
            'cagr_pct': float(cagr * 100)}


def classify_regime(spy_ret, threshold=0.003):
    regimes = pd.Series('flat', index=spy_ret.index)
    regimes[spy_ret > threshold] = 'green'
    regimes[spy_ret < -threshold] = 'red'
    return regimes


def regime_test(strat_rets, spy_rets, name=''):
    regimes = classify_regime(spy_rets.reindex(strat_rets.index))
    results = {}
    for r in ['green', 'red', 'flat']:
        mask = regimes == r
        m = calc_metrics(strat_rets[mask], f'{name} ({r})')
        results[r] = m

    sg = results['green']['sharpe']
    sr = results['red']['sharpe']
    max_s = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / max_s if max_s > 0 else float('inf')

    return {
        'per_regime': results,
        'sharpe_green': float(sg),
        'sharpe_red': float(sr),
        'sharpe_flat': float(results['flat']['sharpe']),
        'regime_gap': float(gap),
        'regime_pass': gap < 0.50,
    }


def download_data(tickers, start=START, end=END):
    data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[t] = df
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    return data


def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ─── CORE: Risk Parity Engine ─────────────────────────────────────────────

def risk_parity_weights(returns_window, leverage=2.0):
    """Inverse-vol risk parity weights, scaled to target leverage."""
    vols = returns_window.std()
    vols = vols.replace(0, np.nan).dropna()
    if len(vols) == 0:
        return pd.Series(0, index=returns_window.columns)
    inv_vol = 1.0 / vols
    w = inv_vol / inv_vol.sum() * leverage
    return w


def run_risk_parity(rets, prices, spy_ret, assets, leverage, tilt_mode='none',
                    vol_target=None, margin_cost_annual=0.0):
    """
    Run walk-forward risk parity with optional tilt and vol targeting.

    tilt_mode: 'none', 'momentum', 'carry', 'meanrev', 'composite', 'vol_target'
    vol_target: if set, scale total portfolio to this annualized vol
    margin_cost_annual: annual cost of leverage (e.g., 0.06 for 6% IBKR margin)
    """
    common_idx = rets.index

    # Pre-compute signals
    mom_12_1 = prices.shift(21) / prices.shift(252) - 1  # 12-1 month momentum
    carry_63 = prices.pct_change(63)  # 3-month carry proxy
    rsi_21 = pd.DataFrame({a: compute_rsi(prices[a], 21) for a in assets})  # 21-day RSI
    realized_vol_63 = rets.rolling(63).std() * np.sqrt(252)  # annualized 63d vol

    strat_rets_list = []
    rebal_dates = list(range(TRAIN_DAYS + 252, len(common_idx) - TEST_DAYS, TEST_DAYS))

    for i in rebal_dates:
        # Base risk parity weights
        lookback = rets[assets].iloc[max(0, i-63):i]
        base_w = risk_parity_weights(lookback, leverage=leverage)

        if tilt_mode == 'none':
            final_w = base_w
        else:
            tilt_scores = pd.Series(0.0, index=assets)

            for a in assets:
                score = 0.0
                n = 0

                if tilt_mode in ['momentum', 'composite']:
                    m = mom_12_1[a].iloc[i] if i < len(mom_12_1) and a in mom_12_1 else np.nan
                    if not np.isnan(m):
                        score += np.clip(m, -0.5, 0.5)
                        n += 1

                if tilt_mode in ['carry', 'composite']:
                    c = carry_63[a].iloc[i] if i < len(carry_63) and a in carry_63 else np.nan
                    if not np.isnan(c):
                        score += np.clip(c, -0.5, 0.5)
                        n += 1

                if tilt_mode in ['meanrev', 'composite']:
                    r = rsi_21[a].iloc[i] if i < len(rsi_21) and a in rsi_21 else np.nan
                    if not np.isnan(r):
                        # Buy oversold (RSI < 30), sell overbought (RSI > 70)
                        score += (50 - r) / 100  # range roughly [-0.5, 0.5]
                        n += 1

                if tilt_mode in ['vol_target', 'composite']:
                    v = realized_vol_63[a].iloc[i] if i < len(realized_vol_63) and a in realized_vol_63 else np.nan
                    if not np.isnan(v) and v > 0:
                        # Inverse vol scaling — already in risk parity, but this adds momentum-vol interaction
                        target_v = 0.15  # target 15% vol per asset
                        score += np.clip((target_v / v - 1) * 0.3, -0.3, 0.3)
                        n += 1

                tilt_scores[a] = score / max(n, 1)

            # Apply tilt as multiplicative factor
            tilt_rank = tilt_scores.rank()
            if tilt_rank.std() > 0:
                tilt_factor = 1.0 + (tilt_rank - tilt_rank.mean()) / tilt_rank.std() * 0.3
            else:
                tilt_factor = pd.Series(1.0, index=assets)
            tilt_factor = tilt_factor.clip(0.5, 1.5)

            final_w = base_w * tilt_factor
            final_w = final_w / final_w.sum() * leverage  # re-normalize

        # Vol targeting: scale entire portfolio to target vol
        if vol_target is not None:
            port_vol = (lookback * base_w).sum(axis=1).std() * np.sqrt(252)
            if port_vol > 0:
                vol_scale = vol_target / port_vol
                vol_scale = np.clip(vol_scale, 0.5, 3.0)  # sanity bounds
                final_w = final_w * vol_scale

        # Apply to next period
        for j in range(i, min(i + TEST_DAYS, len(common_idx) - 1)):
            next_ret = rets[assets].iloc[j + 1] if j + 1 < len(rets) else pd.Series(0, index=assets)
            port_ret = (final_w * next_ret).sum()

            # Subtract margin cost (daily)
            excess_leverage = max(0, final_w.abs().sum() - 1.0)
            daily_margin = margin_cost_annual / 252 * excess_leverage
            port_ret -= daily_margin

            strat_rets_list.append({'date': common_idx[j], 'ret': port_ret})

    strat_s = pd.DataFrame(strat_rets_list).set_index('date')['ret']
    strat_s = strat_s[~strat_s.index.duplicated(keep='first')].dropna()
    return strat_s


# ─── PART A: LEVERAGE SWEEP ───────────────────────────────────────────────

def part_a_leverage_sweep(data):
    print("\n" + "=" * 70)
    print("PART A: LEVERAGE SWEEP (1.5x to 3.5x)")
    print("=" * 70)

    base_assets = ['SPY', 'TLT', 'GLD', 'DBC']
    available = [a for a in base_assets if a in data]

    spy = data['SPY']
    common_idx = spy.index

    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for a in available:
        prices[a] = data[a]['Close'].reindex(common_idx)
        rets[a] = prices[a].pct_change()

    spy_ret = rets['SPY']

    leverage_levels = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    results = []

    for lev in leverage_levels:
        print(f"\n  Testing {lev}x leverage...")
        s = run_risk_parity(rets, prices, spy_ret, available, leverage=lev, tilt_mode='none')
        m = calc_metrics(s, f'RP {lev}x')
        rt = regime_test(s, spy_ret.reindex(s.index).dropna(), f'RP {lev}x')

        result = {
            'leverage': lev,
            'metrics': m,
            'regime': rt,
        }
        results.append(result)

        print(f"    Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, "
              f"Regime gap={rt['regime_gap']:.3f}")

    # Find optimal: maximize CAGR subject to regime_gap < 0.50 and MaxDD > -50%
    viable = [r for r in results if r['regime']['regime_gap'] < 0.50 and r['metrics']['max_dd'] > -0.50]
    if viable:
        best = max(viable, key=lambda r: r['metrics']['cagr'])
        print(f"\n  OPTIMAL LEVERAGE: {best['leverage']}x "
              f"(CAGR={best['metrics']['cagr']:.1%}, MaxDD={best['metrics']['max_dd']:.1%}, "
              f"RegimeGap={best['regime']['regime_gap']:.3f})")
    else:
        best = max(results, key=lambda r: r['metrics']['sharpe'])
        print(f"\n  No viable option (DD/regime), best Sharpe: {best['leverage']}x")

    return results


# ─── PART B: ASSET UNIVERSE EXPANSION ─────────────────────────────────────

def part_b_universe_expansion(data):
    print("\n" + "=" * 70)
    print("PART B: ASSET UNIVERSE EXPANSION")
    print("=" * 70)

    universes = {
        'base_4': ['SPY', 'TLT', 'GLD', 'DBC'],
        'intl_6': ['SPY', 'TLT', 'GLD', 'DBC', 'EFA', 'EEM'],
        'real_7': ['SPY', 'TLT', 'GLD', 'DBC', 'EFA', 'VNQ', 'TIP'],
        'full_8': ['SPY', 'TLT', 'GLD', 'DBC', 'EFA', 'EEM', 'VNQ', 'TIP'],
    }

    spy = data['SPY']
    common_idx = spy.index

    # Build full return/price matrices
    all_tickers = list(set(t for u in universes.values() for t in u))
    all_rets = pd.DataFrame()
    all_prices = pd.DataFrame()
    for t in all_tickers:
        if t in data:
            all_prices[t] = data[t]['Close'].reindex(common_idx)
            all_rets[t] = all_prices[t].pct_change()

    spy_ret = all_rets['SPY']

    results = {}
    leverage = 2.0  # use 2x as baseline for comparison

    for name, universe in universes.items():
        avail = [a for a in universe if a in all_rets.columns]
        if len(avail) < 3:
            print(f"  {name}: skipping, only {len(avail)} assets available")
            continue

        print(f"\n  Testing {name}: {avail}")
        s = run_risk_parity(all_rets, all_prices, spy_ret, avail, leverage=leverage, tilt_mode='none')
        m = calc_metrics(s, f'{name} RP {leverage}x')
        rt = regime_test(s, spy_ret.reindex(s.index).dropna(), name)

        results[name] = {
            'assets': avail,
            'n_assets': len(avail),
            'metrics': m,
            'regime': rt,
        }

        print(f"    Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, "
              f"Regime gap={rt['regime_gap']:.3f}")

    # Find best universe
    if results:
        best_name = max(results, key=lambda k: results[k]['metrics']['sharpe'])
        print(f"\n  BEST UNIVERSE: {best_name} ({results[best_name]['assets']})")

    return results


# ─── PART C: SMARTER PREDICTIVE TILTS ─────────────────────────────────────

def part_c_predictive_tilts(data):
    print("\n" + "=" * 70)
    print("PART C: SMARTER PREDICTIVE TILTS")
    print("=" * 70)

    base_assets = ['SPY', 'TLT', 'GLD', 'DBC']
    available = [a for a in base_assets if a in data]

    spy = data['SPY']
    common_idx = spy.index

    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for a in available:
        prices[a] = data[a]['Close'].reindex(common_idx)
        rets[a] = prices[a].pct_change()

    spy_ret = rets['SPY']
    leverage = 2.0

    tilt_modes = ['none', 'momentum', 'carry', 'meanrev', 'vol_target', 'composite']
    results = {}

    for mode in tilt_modes:
        print(f"\n  Testing tilt: {mode}...")
        s = run_risk_parity(rets, prices, spy_ret, available, leverage=leverage, tilt_mode=mode)
        m = calc_metrics(s, f'RP 2x + {mode}')
        rt = regime_test(s, spy_ret.reindex(s.index).dropna(), f'{mode}')

        results[mode] = {
            'tilt_mode': mode,
            'metrics': m,
            'regime': rt,
        }

        print(f"    Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
              f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, "
              f"Regime gap={rt['regime_gap']:.3f}")

    # Also test vol targeting overlay
    print(f"\n  Testing vol_target overlay (15% target)...")
    s = run_risk_parity(rets, prices, spy_ret, available, leverage=leverage,
                        tilt_mode='composite', vol_target=0.15)
    m = calc_metrics(s, 'RP 2x + composite + vol_target')
    rt = regime_test(s, spy_ret.reindex(s.index).dropna(), 'composite+voltgt')
    results['composite_voltgt'] = {
        'tilt_mode': 'composite + vol_target(15%)',
        'metrics': m,
        'regime': rt,
    }
    print(f"    Sharpe={m['sharpe']:.3f}, Sortino={m['sortino']:.3f}, "
          f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, "
          f"Regime gap={rt['regime_gap']:.3f}")

    # Find best tilt
    best_mode = max(results, key=lambda k: results[k]['metrics']['sharpe'])
    print(f"\n  BEST TILT: {best_mode} (Sharpe={results[best_mode]['metrics']['sharpe']:.3f})")

    return results


# ─── PART D: IMPLEMENTATION — MARGIN vs LEVERAGED ETFs ────────────────────

def part_d_implementation(data):
    print("\n" + "=" * 70)
    print("PART D: IMPLEMENTATION — MARGIN vs LEVERAGED ETFs")
    print("=" * 70)

    spy = data['SPY']
    common_idx = spy.index
    spy_ret = spy['Close'].reindex(common_idx).pct_change()

    # Version A: Unleveraged ETFs + IBKR margin (6% annual cost on borrowed)
    print("\n  Version A: Regular ETFs + IBKR margin at 6%...")
    base_assets = ['SPY', 'TLT', 'GLD', 'DBC']
    available = [a for a in base_assets if a in data]

    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for a in available:
        prices[a] = data[a]['Close'].reindex(common_idx)
        rets[a] = prices[a].pct_change()

    s_margin = run_risk_parity(rets, prices, spy_ret, available, leverage=2.0,
                               tilt_mode='composite', margin_cost_annual=0.06)
    m_margin = calc_metrics(s_margin, 'RP 2x + Margin(6%)')
    rt_margin = regime_test(s_margin, spy_ret.reindex(s_margin.index).dropna(), 'margin')

    print(f"    Sharpe={m_margin['sharpe']:.3f}, CAGR={m_margin['cagr']:.1%}, "
          f"MaxDD={m_margin['max_dd']:.1%}, Regime gap={rt_margin['regime_gap']:.3f}")

    # Version B: Leveraged ETFs (with vol drag approximation)
    # Leveraged ETFs: daily reset causes vol drag. Approximate:
    # drag ~ leverage * (leverage - 1) * sigma^2 / 2 per day (continuous approx)
    print("\n  Version B: Leveraged ETFs (UPRO/TMF/UGL) with vol drag...")

    lev_map = {
        'SPY': ('UPRO', 3.0),   # 3x SPY
        'TLT': ('TMF', 3.0),    # 3x TLT
        'GLD': ('UGL', 2.0),    # 2x GLD (no 3x gold ETF with good liquidity)
    }

    # Simulate leveraged ETF returns WITH vol drag
    def sim_lev_etf_returns(base_rets, lev_factor):
        """Simulate leveraged ETF daily returns including vol drag."""
        # Daily return of leveraged ETF: lev * base_ret
        # But vol drag: over time, E[lev_etf] < lev * E[base]
        # Daily drag ~ -0.5 * lev * (lev-1) * variance
        daily_var = base_rets.rolling(63).var()
        drag = 0.5 * lev_factor * (lev_factor - 1) * daily_var
        lev_ret = lev_factor * base_rets - drag
        return lev_ret

    lev_rets = pd.DataFrame()
    for base, (lev_ticker, lev_factor) in lev_map.items():
        if base in rets:
            lev_rets[lev_ticker] = sim_lev_etf_returns(rets[base], lev_factor)

    # For DBC, no good leveraged ETF, use 1x
    if 'DBC' in rets:
        lev_rets['DBC'] = rets['DBC']

    lev_assets = list(lev_rets.columns)
    lev_prices = (1 + lev_rets).cumprod() * 100  # synthetic price series

    # Run RP on leveraged ETFs at 1x (leverage is already in the ETFs)
    s_levETF = run_risk_parity(lev_rets, lev_prices, spy_ret, lev_assets,
                               leverage=1.0, tilt_mode='none')
    m_levETF = calc_metrics(s_levETF, 'Leveraged ETFs RP 1x')
    rt_levETF = regime_test(s_levETF, spy_ret.reindex(s_levETF.index).dropna(), 'lev_etf')

    print(f"    Sharpe={m_levETF['sharpe']:.3f}, CAGR={m_levETF['cagr']:.1%}, "
          f"MaxDD={m_levETF['max_dd']:.1%}, Regime gap={rt_levETF['regime_gap']:.3f}")

    # Compare
    margin_better = m_margin['sharpe'] > m_levETF['sharpe']
    print(f"\n  WINNER: {'Margin (Version A)' if margin_better else 'Leveraged ETFs (Version B)'}")
    print(f"  Margin approach preserves Sharpe better (no vol drag) but has carrying cost.")
    print(f"  Leveraged ETFs have no margin cost but suffer vol drag in choppy markets.")

    return {
        'margin': {'metrics': m_margin, 'regime': rt_margin},
        'lev_etf': {'metrics': m_levETF, 'regime': rt_levETF},
        'margin_better': margin_better,
    }


# ─── COMBINED OPTIMAL: Best universe + best tilt + best leverage ───────────

def combined_optimal(data):
    print("\n" + "=" * 70)
    print("COMBINED OPTIMAL: BEST OF EVERYTHING")
    print("=" * 70)

    # Use expanded universe
    full_assets = ['SPY', 'TLT', 'GLD', 'DBC', 'EFA', 'EEM', 'VNQ', 'TIP']
    available = [a for a in full_assets if a in data]

    spy = data['SPY']
    common_idx = spy.index

    rets = pd.DataFrame()
    prices = pd.DataFrame()
    for a in available:
        prices[a] = data[a]['Close'].reindex(common_idx)
        rets[a] = prices[a].pct_change()

    spy_ret = rets['SPY']

    # Test grid: leverage x tilt
    combos = []
    for lev in [2.0, 2.5, 3.0]:
        for tilt in ['none', 'composite']:
            for vt in [None, 0.12, 0.15, 0.20]:
                label = f'L{lev}_{tilt}_vt{vt}'
                print(f"  Testing {label}...")
                s = run_risk_parity(rets, prices, spy_ret, available,
                                    leverage=lev, tilt_mode=tilt, vol_target=vt,
                                    margin_cost_annual=0.06)
                m = calc_metrics(s, label)
                rt = regime_test(s, spy_ret.reindex(s.index).dropna(), label)

                combos.append({
                    'label': label,
                    'leverage': lev,
                    'tilt': tilt,
                    'vol_target': vt,
                    'metrics': m,
                    'regime': rt,
                })

    # Sort by Sharpe
    combos.sort(key=lambda x: x['metrics']['sharpe'], reverse=True)

    print(f"\n  TOP 5 CONFIGURATIONS:")
    for i, c in enumerate(combos[:5]):
        m = c['metrics']
        rt = c['regime']
        print(f"    {i+1}. {c['label']}: Sharpe={m['sharpe']:.3f}, "
              f"CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, "
              f"RegimeGap={rt['regime_gap']:.3f}")

    return combos


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R5 TASK 1: MAXIMIZE LEVERAGED RISK PARITY")
    print(f"Start: {START} | End: {END}")
    print(f"Walk-forward: {TRAIN_DAYS}d train, {TEST_DAYS}d test (sliding)")
    print("=" * 70)

    # Download all needed data
    all_tickers = ['SPY', 'TLT', 'GLD', 'DBC', 'EFA', 'EEM', 'VNQ', 'TIP', 'DBMF']
    print("\nDownloading data...")
    data = download_data(all_tickers)

    all_results = {}

    # Part A: Leverage sweep
    all_results['part_a_leverage_sweep'] = part_a_leverage_sweep(data)

    # Part B: Universe expansion
    all_results['part_b_universe'] = part_b_universe_expansion(data)

    # Part C: Predictive tilts
    all_results['part_c_tilts'] = part_c_predictive_tilts(data)

    # Part D: Implementation
    all_results['part_d_implementation'] = part_d_implementation(data)

    # Combined optimal
    all_results['combined_optimal'] = combined_optimal(data)

    # ─── FINAL SUMMARY ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Best from combined
    if all_results['combined_optimal']:
        best = all_results['combined_optimal'][0]
        m = best['metrics']
        rt = best['regime']
        print(f"\n  BEST OVERALL CONFIG: {best['label']}")
        print(f"  Leverage: {best['leverage']}x | Tilt: {best['tilt']} | Vol Target: {best['vol_target']}")
        print(f"  Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f}")
        print(f"  CAGR: {m['cagr']:.1%} | MaxDD: {m['max_dd']:.1%}")
        print(f"  Win Rate: {m['win_rate']:.1%} | Profit Factor: {m['pf']:.2f}")
        print(f"  Regime Gap: {rt['regime_gap']:.3f} ({'PASS' if rt['regime_pass'] else 'FAIL'})")
        print(f"    Green Sharpe: {rt['sharpe_green']:.3f} | Red Sharpe: {rt['sharpe_red']:.3f}")

    # Save results
    # Convert to serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    out_file = os.path.join(OUT_DIR, 'task1_lev_risk_parity_results.json')
    with open(out_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")


if __name__ == '__main__':
    main()
