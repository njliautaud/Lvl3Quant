#!/usr/bin/env python3
"""
R6 TASK 1: RISK PARITY (3x) + OPTIONS INCOME OVERLAY

Can we add 5-10% CAGR from options premium income on top of the 3x risk parity base?

Sub-analyses:
A) Covered calls on UPRO: sell monthly ~30-delta calls against leveraged equity position
B) Cash-secured puts on pullbacks: sell puts when base assets pull back >5%
C) Combined overlay: calls + puts together
D) Walk-forward regime test: does the combined portfolio maintain regime agnosticism?

HC #694: Commission-free (Robinhood)
HC #428: Regime test (report honestly)
HC #0: Sliding window walk-forward
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime
from scipy.stats import norm

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r6'
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
    wr = (rets > 0).mean()
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    return {'name': name, 'sharpe': round(float(sharpe), 4),
            'sortino': round(float(sortino), 4),
            'cagr': round(float(cagr), 4), 'cagr_pct': round(float(cagr*100), 2),
            'max_dd': round(float(max_dd), 4), 'max_dd_pct': round(float(max_dd*100), 2),
            'win_rate': round(float(wr), 4), 'pf': round(float(pf), 3),
            'calmar': round(float(calmar), 3),
            'n_days': int(len(rets))}


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
        'sharpe_green': round(float(sg), 4),
        'sharpe_red': round(float(sr), 4),
        'regime_gap': round(float(gap), 4),
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


# ─── Black-Scholes for options pricing ────────────────────────────────────

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(0, K - S)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


def find_strike_by_delta(S, T, r, sigma, target_delta, option_type='call',
                         precision=0.01):
    """Find strike price that gives target delta."""
    # Binary search
    lo, hi = S * 0.5, S * 2.0
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma, option_type)
        if option_type == 'call':
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            if abs(d) > abs(target_delta):
                hi = mid
            else:
                lo = mid
        if hi - lo < precision:
            break
    return (lo + hi) / 2


# ─── Risk Parity Engine (from R5) ─────────────────────────────────────────

def risk_parity_weights(returns_window, leverage=1.0):
    vols = returns_window.std()
    vols = vols.replace(0, np.nan).dropna()
    if len(vols) == 0:
        return pd.Series(0, index=returns_window.columns)
    inv_vol = 1.0 / vols
    w = inv_vol / inv_vol.sum() * leverage
    return w


def run_risk_parity_base(rets, prices, assets, leverage=3.0):
    """Run base risk parity, return daily returns series."""
    common_idx = rets.index
    mom_12_1 = prices.shift(21) / prices.shift(252) - 1

    strat_rets_list = []
    rebal_dates = list(range(TRAIN_DAYS + 252, len(common_idx) - TEST_DAYS, TEST_DAYS))

    for i in rebal_dates:
        lookback = rets[assets].iloc[max(0, i-63):i]
        base_w = risk_parity_weights(lookback, leverage=leverage)

        # Momentum tilt
        tilt_scores = {}
        for a in assets:
            m = mom_12_1[a].iloc[i] if i < len(mom_12_1) and a in mom_12_1 else np.nan
            tilt_scores[a] = np.clip(m, -0.5, 0.5) if not np.isnan(m) else 0.0

        tilt_series = pd.Series(tilt_scores)
        tilt_rank = tilt_series.rank()
        if tilt_rank.std() > 0:
            tilt_factor = 1.0 + (tilt_rank - tilt_rank.mean()) / tilt_rank.std() * 0.3
        else:
            tilt_factor = pd.Series(1.0, index=assets)
        tilt_factor = tilt_factor.clip(0.5, 1.5)

        final_w = base_w * tilt_factor
        final_w = final_w / final_w.sum() * leverage

        for j in range(i, min(i + TEST_DAYS, len(common_idx) - 1)):
            next_ret = rets[assets].iloc[j + 1] if j + 1 < len(rets) else pd.Series(0, index=assets)
            port_ret = (final_w * next_ret).sum()
            strat_rets_list.append({
                'date': common_idx[j],
                'ret': port_ret,
                'spy_weight': final_w.get('SPY', 0),
            })

    df = pd.DataFrame(strat_rets_list).set_index('date')
    df = df[~df.index.duplicated(keep='first')]
    return df


# ─── COVERED CALL OVERLAY ─────────────────────────────────────────────────

def simulate_covered_call_overlay(base_df, spy_prices, spy_vols, call_delta=0.30,
                                  expiry_days=30, overlay_pct=1.0):
    """
    Simulate selling covered calls on the equity portion of the portfolio.

    call_delta: delta of the call to sell (0.30 = ~30 delta OTM call)
    expiry_days: days to expiry for each call cycle
    overlay_pct: fraction of equity position to overlay
    """
    results = []
    r = 0.05  # risk-free rate

    # Roll calls monthly
    cycle_days = 0
    call_strike = None
    premium_collected = 0

    for idx in base_df.index:
        if idx not in spy_prices.index:
            continue

        S = float(spy_prices.loc[idx])
        vol_idx = spy_vols.index.get_indexer([idx], method='ffill')[0]
        sigma = float(spy_vols.iloc[vol_idx]) if vol_idx >= 0 else 0.20

        base_ret = float(base_df.loc[idx, 'ret'])
        spy_w = float(base_df.loc[idx, 'spy_weight'])

        # Start new call cycle
        if cycle_days <= 0 or call_strike is None:
            T = expiry_days / 252
            call_strike = find_strike_by_delta(S, T, r, sigma, call_delta, 'call')
            premium = bs_call_price(S, call_strike, T, r, sigma)
            premium_pct = premium / S * overlay_pct * abs(spy_w)
            premium_collected = premium_pct
            cycle_days = expiry_days

        cycle_days -= 1

        # At expiry or month end
        if cycle_days <= 0:
            # Settlement
            T_rem = 1 / 252
            call_value = max(0, S - call_strike)
            assignment_cost = call_value / S * overlay_pct * abs(spy_w)

            # Net income = premium - assignment cost, amortized over the cycle
            daily_income = (premium_collected - assignment_cost) / expiry_days
        else:
            # Mark-to-market: option decays (theta income)
            T_rem = cycle_days / 252
            call_value = bs_call_price(S, call_strike, T_rem, r, sigma)
            # Rough daily theta income
            daily_income = premium_collected / expiry_days * 0.3  # ~30% is realized as daily theta

        overlay_ret = base_ret + daily_income
        results.append({'date': idx, 'ret': overlay_ret, 'base_ret': base_ret,
                        'overlay_income': daily_income})

    df = pd.DataFrame(results).set_index('date')
    return df['ret']


# ─── PUT SELLING ON PULLBACKS ──────────────────────────────────────────────

def simulate_put_selling_overlay(base_df, spy_prices, spy_vols, put_delta=-0.25,
                                 pullback_threshold=-0.05, expiry_days=30):
    """
    Sell puts on SPY/UPRO when there's been a pullback (>5% decline in 20 days).
    Premium income adds to base portfolio return.
    """
    results = []
    r = 0.05

    # Track pullback state
    spy_ret_20d = spy_prices.pct_change(20)

    cycle_days = 0
    put_strike = None
    premium_collected = 0
    put_active = False

    for idx in base_df.index:
        if idx not in spy_prices.index or idx not in spy_ret_20d.index:
            continue

        S = float(spy_prices.loc[idx])
        ret_20d = float(spy_ret_20d.loc[idx]) if not np.isnan(spy_ret_20d.loc[idx]) else 0

        vol_idx = spy_vols.index.get_indexer([idx], method='ffill')[0]
        sigma = float(spy_vols.iloc[vol_idx]) if vol_idx >= 0 else 0.20

        base_ret = float(base_df.loc[idx, 'ret'])

        # Check if pullback triggers put selling
        if cycle_days <= 0:
            if ret_20d < pullback_threshold:
                T = expiry_days / 252
                put_strike = find_strike_by_delta(S, T, r, sigma, put_delta, 'put')
                premium = bs_put_price(S, put_strike, T, r, sigma)
                # Size: 20% of portfolio notional
                premium_pct = premium / S * 0.20
                premium_collected = premium_pct
                cycle_days = expiry_days
                put_active = True
            else:
                put_active = False

        if put_active and cycle_days > 0:
            cycle_days -= 1

            if cycle_days <= 0:
                # Settlement
                put_value = max(0, put_strike - S)
                assignment_cost = put_value / S * 0.20
                daily_income = (premium_collected - assignment_cost) / expiry_days
                put_active = False
            else:
                daily_income = premium_collected / expiry_days * 0.3
        else:
            daily_income = 0

        overlay_ret = base_ret + daily_income
        results.append({'date': idx, 'ret': overlay_ret, 'base_ret': base_ret,
                        'overlay_income': daily_income, 'pullback_20d': ret_20d})

    df = pd.DataFrame(results).set_index('date')
    return df['ret']


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R6 TASK 1: RISK PARITY (3x) + OPTIONS INCOME OVERLAY")
    print(f"Period: {START} to {END}")
    print("=" * 70)

    # Download data
    tickers = ['SPY', 'TLT', 'GLD', 'DBC', '^VIX']
    print("\nDownloading data...")
    data = download_data(tickers)

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
    spy_prices = prices['SPY']
    spy_vols = rets['SPY'].rolling(63).std() * np.sqrt(252)

    all_results = {}

    # ─── Part A: Base Risk Parity 3x ───────────────────────────────────
    print("\n" + "=" * 60)
    print("PART A: BASE 3x RISK PARITY + MOMENTUM TILT (BASELINE)")
    print("=" * 60)

    base_df = run_risk_parity_base(rets, prices, available, leverage=3.0)
    base_rets = base_df['ret']
    base_metrics = calc_metrics(base_rets, 'Base RP 3x')
    base_regime = regime_test(base_rets, spy_ret.reindex(base_rets.index).dropna(), 'Base RP 3x')

    print(f"  Sharpe={base_metrics['sharpe']}, CAGR={base_metrics['cagr_pct']}%, "
          f"MaxDD={base_metrics['max_dd_pct']}%, Calmar={base_metrics['calmar']}")
    print(f"  Regime gap={base_regime['regime_gap']}, "
          f"Green={base_regime['sharpe_green']}, Red={base_regime['sharpe_red']}")

    all_results['base_rp3x'] = {'metrics': base_metrics, 'regime': base_regime}

    # ─── Part B: Covered Call Overlay ──────────────────────────────────
    print("\n" + "=" * 60)
    print("PART B: COVERED CALL OVERLAY ON UPRO")
    print("=" * 60)

    for delta in [0.20, 0.30, 0.40]:
        label = f'CC_delta{int(delta*100)}'
        print(f"\n  Testing {label}...")
        cc_rets = simulate_covered_call_overlay(
            base_df, spy_prices, spy_vols, call_delta=delta
        )
        cc_metrics = calc_metrics(cc_rets, label)
        cc_regime = regime_test(cc_rets, spy_ret.reindex(cc_rets.index).dropna(), label)

        print(f"    Sharpe={cc_metrics['sharpe']}, CAGR={cc_metrics['cagr_pct']}%, "
              f"MaxDD={cc_metrics['max_dd_pct']}%, Calmar={cc_metrics['calmar']}")
        print(f"    Regime gap={cc_regime['regime_gap']}")

        added_cagr = cc_metrics['cagr_pct'] - base_metrics['cagr_pct']
        print(f"    Added CAGR from calls: {added_cagr:+.2f}%")

        all_results[label] = {'metrics': cc_metrics, 'regime': cc_regime,
                              'added_cagr': round(added_cagr, 2)}

    # ─── Part C: Put Selling on Pullbacks ──────────────────────────────
    print("\n" + "=" * 60)
    print("PART C: PUT SELLING ON PULLBACKS")
    print("=" * 60)

    for pullback in [-0.03, -0.05, -0.08]:
        label = f'Puts_pb{int(abs(pullback)*100)}pct'
        print(f"\n  Testing {label}...")
        put_rets = simulate_put_selling_overlay(
            base_df, spy_prices, spy_vols, pullback_threshold=pullback
        )
        put_metrics = calc_metrics(put_rets, label)
        put_regime = regime_test(put_rets, spy_ret.reindex(put_rets.index).dropna(), label)

        print(f"    Sharpe={put_metrics['sharpe']}, CAGR={put_metrics['cagr_pct']}%, "
              f"MaxDD={put_metrics['max_dd_pct']}%, Calmar={put_metrics['calmar']}")
        print(f"    Regime gap={put_regime['regime_gap']}")

        added_cagr = put_metrics['cagr_pct'] - base_metrics['cagr_pct']
        print(f"    Added CAGR from puts: {added_cagr:+.2f}%")

        all_results[label] = {'metrics': put_metrics, 'regime': put_regime,
                              'added_cagr': round(added_cagr, 2)}

    # ─── Part D: Combined (Calls + Puts) ──────────────────────────────
    print("\n" + "=" * 60)
    print("PART D: COMBINED OVERLAY (CALLS + PUTS)")
    print("=" * 60)

    # Best call + best put combination
    # Use 30-delta calls + 5% pullback puts
    cc_rets_best = simulate_covered_call_overlay(
        base_df, spy_prices, spy_vols, call_delta=0.30
    )
    # Re-run with calls already in, then add puts
    # Approximate: add both overlays independently
    cc_income = cc_rets_best - base_rets.reindex(cc_rets_best.index)
    put_rets_best = simulate_put_selling_overlay(
        base_df, spy_prices, spy_vols, pullback_threshold=-0.05
    )
    put_income = put_rets_best - base_rets.reindex(put_rets_best.index)

    # Combined
    common = base_rets.index.intersection(cc_income.index).intersection(put_income.index)
    combined_rets = base_rets.loc[common] + cc_income.loc[common] + put_income.loc[common]

    combined_metrics = calc_metrics(combined_rets, 'Combined Overlay')
    combined_regime = regime_test(combined_rets, spy_ret.reindex(combined_rets.index).dropna(),
                                 'Combined')

    print(f"  Sharpe={combined_metrics['sharpe']}, CAGR={combined_metrics['cagr_pct']}%, "
          f"MaxDD={combined_metrics['max_dd_pct']}%, Calmar={combined_metrics['calmar']}")
    print(f"  Regime gap={combined_regime['regime_gap']}")

    added_cagr = combined_metrics['cagr_pct'] - base_metrics['cagr_pct']
    print(f"  Added CAGR from combined overlay: {added_cagr:+.2f}%")

    all_results['combined_overlay'] = {
        'metrics': combined_metrics, 'regime': combined_regime,
        'added_cagr': round(added_cagr, 2),
    }

    # ─── FINAL SUMMARY ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  {'Strategy':<30} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Gap':>8} {'Added':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for key, val in all_results.items():
        m = val['metrics']
        rt = val['regime']
        added = val.get('added_cagr', 0)
        print(f"  {key:<30} {m['cagr_pct']:>7.1f}% {m['sharpe']:>8.3f} "
              f"{m['max_dd_pct']:>7.1f}% {rt['regime_gap']:>8.3f} {added:>+7.1f}%")

    # Save results
    out_file = os.path.join(OUT_DIR, 'task1_riskparity_income_overlay_results.json')
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")


if __name__ == '__main__':
    main()
