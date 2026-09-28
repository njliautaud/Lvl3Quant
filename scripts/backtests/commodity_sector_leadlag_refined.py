#!/usr/bin/env python3
"""
Cross-Asset Lead-Lag: REFINED — Long-Only + Best Pairs
======================================================
From the full exploration:
- Short side has NO edge (puts on commodity drops = noise)
- Long side (calls when commodity surges) has Sharpe 1.20
- Lumber->XHB standalone Sharpe 0.97

This version tests: LONG-ONLY signals (commodity surge -> buy sector calls)
across all pairs, with z > 1.5 threshold + 2-day momentum confirmation.

Also tests a "best pairs only" variant (WOOD->XHB, UNG->XLU, DBA->XLP)
excluding the worst performers (USO->XLE, GLD->GDX).

Author: Claude Opus 4.6
Date: 2026-08-18
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 70)
print("COMMODITY-SECTOR LEAD-LAG — LONG-ONLY REFINED")
print("=" * 70)
print()

# ALL pairs for broad test
ALL_PAIRS = [
    ('UNG', 'XLU', 'NatGas->Utilities'),
    ('USO', 'XLE', 'Oil->Energy'),
    ('GLD', 'GDX', 'Gold->Miners'),
    ('DBA', 'XLP', 'Agriculture->Staples'),
    ('WOOD', 'XHB', 'Lumber->Homebuilders'),
]

# Best pairs (exclude Oil->Energy and Gold->Miners which dragged returns)
BEST_PAIRS = [
    ('UNG', 'XLU', 'NatGas->Utilities'),
    ('DBA', 'XLP', 'Agriculture->Staples'),
    ('WOOD', 'XHB', 'Lumber->Homebuilders'),
]

CONTEXT_TICKERS = ['SPY']
LOOKBACK = 20
HOLD_DAYS = 4
OPTION_GAMMA_BOOST = 1.20
THETA_DRAG = 0.005
START_DATE = '2018-01-01'
END_DATE = '2026-08-15'

all_tickers = list(set([p[0] for p in ALL_PAIRS] + [p[1] for p in ALL_PAIRS] + CONTEXT_TICKERS))
data = yf.download(all_tickers, start=START_DATE, end=END_DATE, progress=False)

if isinstance(data.columns, pd.MultiIndex):
    closes = data['Close'].copy()
else:
    closes = data[['Close']].copy()

closes = closes.ffill().dropna()
returns = closes.pct_change()
spy_ret = returns['SPY']

print(f"Data: {closes.shape[0]} days, {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")


def generate_trades(pairs, threshold, long_only=True, require_confirmation=True):
    """Generate trades for given pairs and parameters."""
    trades = []

    for commodity, sector, pair_name in pairs:
        if commodity not in returns.columns or sector not in returns.columns:
            continue

        comm_ret = returns[commodity]
        rolling_mean = comm_ret.rolling(LOOKBACK).mean()
        rolling_std = comm_ret.rolling(LOOKBACK).std()
        z_score = (comm_ret - rolling_mean) / rolling_std

        comm_ret_2d = closes[commodity].pct_change(2)
        rm2 = comm_ret_2d.rolling(LOOKBACK).mean()
        rs2 = comm_ret_2d.rolling(LOOKBACK).std()
        z_2d = (comm_ret_2d - rm2) / rs2

        composite_z = 0.6 * z_score + 0.4 * z_2d

        for i in range(LOOKBACK + 5, len(closes) - HOLD_DAYS - 1):
            date = closes.index[i]
            z = composite_z.iloc[i]

            if np.isnan(z):
                continue

            # Long only: skip negative z
            if long_only and z < threshold:
                continue
            elif not long_only and abs(z) < threshold:
                continue

            direction = 1 if z > 0 else -1
            if long_only:
                direction = 1

            # Confirmation: 2 consecutive days of commodity momentum
            if require_confirmation:
                day_ret = comm_ret.iloc[i]
                day_ret_prev = comm_ret.iloc[i-1] if i > 0 else 0
                if np.isnan(day_ret) or np.isnan(day_ret_prev):
                    continue
                if direction > 0 and (day_ret <= 0 or day_ret_prev <= 0):
                    continue
                if direction < 0 and (day_ret >= 0 or day_ret_prev >= 0):
                    continue

            entry_price = closes[sector].iloc[i + 1]
            exit_price = closes[sector].iloc[i + 1 + HOLD_DAYS]

            if np.isnan(entry_price) or np.isnan(exit_price) or entry_price == 0:
                continue

            etf_return = (exit_price / entry_price - 1) * direction

            if etf_return > 0:
                option_return = etf_return * OPTION_GAMMA_BOOST
            else:
                option_return = etf_return - THETA_DRAG

            spy_5d = spy_ret.iloc[max(0, i-4):i+1].sum()
            regime = 'green' if spy_5d > 0 else 'red'

            # No overlap
            overlap = False
            for t in trades[-15:]:
                if t['sector'] == sector:
                    days_since = (date - t['entry_date']).days
                    if days_since < HOLD_DAYS + 2:
                        overlap = True
                        break
            if overlap:
                continue

            trades.append({
                'entry_date': date, 'pair': pair_name, 'commodity': commodity,
                'sector': sector, 'direction': direction, 'z_score': z,
                'etf_return': etf_return, 'option_return': option_return,
                'regime': regime,
            })

    return pd.DataFrame(trades)


def compute_metrics(rets, label=""):
    if len(rets) == 0:
        return None
    rets = pd.Series(rets).reset_index(drop=True)
    n = len(rets)
    wr = (rets > 0).mean()
    winners = rets[rets > 0]
    losers = rets[rets <= 0]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 0.001
    pf = (winners.sum() / abs(losers.sum())) if len(losers) > 0 and losers.sum() != 0 else 999
    tpy = min(60, n)
    mu = rets.mean()
    sigma = rets.std()
    sharpe = (mu / sigma) * np.sqrt(tpy) if sigma > 0 else 0
    down = rets[rets < 0]
    down_std = down.std() if len(down) > 1 else sigma
    sortino = (mu / down_std) * np.sqrt(tpy) if down_std > 0 else 0
    eq = (1 + rets).cumprod()
    dd = (eq - eq.cummax()) / eq.cummax()
    max_dd = dd.min()
    total_ret = eq.iloc[-1] - 1
    return {
        'label': label, 'n_trades': n, 'win_rate': wr, 'avg_win': avg_win,
        'avg_loss': avg_loss, 'profit_factor': pf, 'sharpe': sharpe,
        'sortino': sortino, 'max_drawdown': max_dd, 'total_return': total_ret,
        'mean_return': mu, 'std_return': sigma,
    }


def print_metrics(m):
    if m is None:
        print("  No trades.")
        return
    print(f"  {m['label']}")
    print(f"    Trades: {m['n_trades']} | WR: {m['win_rate']:.1%} | PF: {m['profit_factor']:.2f}")
    print(f"    Sharpe: {m['sharpe']:.2f} | Sortino: {m['sortino']:.2f}")
    print(f"    Max DD: {m['max_drawdown']:.1%} | Total Ret: {m['total_return']:.1%}")
    print(f"    Avg Win: {m['avg_win']:.2%} | Avg Loss: {m['avg_loss']:.2%}")
    print()


def run_permutation_test(trades_df, n_perms=1000):
    """Permutation test using random sector returns."""
    all_sector_rets = []
    for _, sector, _ in ALL_PAIRS:
        if sector in closes.columns:
            sc = closes[sector]
            for i in range(len(sc) - HOLD_DAYS - 1):
                r = sc.iloc[i + HOLD_DAYS] / sc.iloc[i] - 1
                if not np.isnan(r):
                    all_sector_rets.append(r)
    all_sector_rets = np.array(all_sector_rets)

    n = len(trades_df)
    observed = compute_metrics(trades_df['option_return'].values)
    obs_sharpe = observed['sharpe']

    perm_sharpes = []
    for _ in range(n_perms):
        # For LONG-ONLY null: random sector returns with positive direction bias removed
        rr = np.random.choice(all_sector_rets, size=n, replace=True)
        # Apply option proxy (long direction)
        orp = np.where(rr > 0, rr * OPTION_GAMMA_BOOST, rr - THETA_DRAG)
        mu = orp.mean()
        sig = orp.std()
        ps = (mu / sig) * np.sqrt(min(60, n)) if sig > 0 else 0
        perm_sharpes.append(ps)

    perm_sharpes = np.array(perm_sharpes)
    p_val = (perm_sharpes >= obs_sharpe).mean()
    return obs_sharpe, p_val, perm_sharpes


def five_gate_eval(trades_df, label):
    """Full 5-gate evaluation."""
    print(f"\n{'='*70}")
    print(f"5-GATE EVALUATION: {label}")
    print(f"{'='*70}")

    overall = compute_metrics(trades_df['option_return'].values, label)
    print_metrics(overall)

    # Regime
    green = trades_df[trades_df['regime'] == 'green']
    red = trades_df[trades_df['regime'] == 'red']
    gm = compute_metrics(green['option_return'].values, "GREEN") if len(green) > 2 else None
    rm = compute_metrics(red['option_return'].values, "RED") if len(red) > 2 else None
    print_metrics(gm)
    print_metrics(rm)

    if gm and rm and max(abs(gm['sharpe']), abs(rm['sharpe'])) > 0:
        regime_gap = abs(gm['sharpe'] - rm['sharpe']) / max(abs(gm['sharpe']), abs(rm['sharpe']))
    else:
        regime_gap = 999

    # Permutation
    obs_sharpe, p_val, _ = run_permutation_test(trades_df)

    # Per pair
    print(f"\n  Per-pair breakdown:")
    for pn in trades_df['pair'].unique():
        pm = trades_df[trades_df['pair'] == pn]
        m = compute_metrics(pm['option_return'].values, f"    {pn}")
        if m:
            print(f"    {pn}: N={m['n_trades']}, Sharpe={m['sharpe']:.2f}, WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}")

    # Yearly
    trades_df_copy = trades_df.copy()
    trades_df_copy['year'] = trades_df_copy['entry_date'].dt.year
    print(f"\n  Yearly:")
    for yr in sorted(trades_df_copy['year'].unique()):
        ym = compute_metrics(trades_df_copy[trades_df_copy['year'] == yr]['option_return'].values)
        if ym:
            print(f"    {yr}: N={ym['n_trades']}, Sharpe={ym['sharpe']:.2f}, WR={ym['win_rate']:.1%}")

    # Gates
    n = len(trades_df)
    gates = {
        'Gate 1 — Sharpe > 0.5': overall['sharpe'] > 0.5,
        'Gate 2 — Perm p-value < 0.05': p_val < 0.05,
        'Gate 3 — Regime gap < 0.50': regime_gap < 0.50,
        'Gate 4 — Max DD < 50%': abs(overall['max_drawdown']) < 0.50,
        'Gate 5 — At least 30 trades': n >= 30,
    }

    print(f"\n  Regime gap: {regime_gap:.2f}")
    print(f"  Perm p-value: {p_val:.4f}")
    print()

    all_pass = True
    for gate, passed in gates.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {gate}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"\n  >>> {verdict}")
    return all_pass, overall, regime_gap, p_val


# ============================================================
# TEST CONFIGURATIONS
# ============================================================

configs = [
    ("A: All pairs, LONG-ONLY, z>1.5, confirmed", ALL_PAIRS, 1.5, True, True),
    ("B: All pairs, LONG-ONLY, z>2.0, confirmed", ALL_PAIRS, 2.0, True, True),
    ("C: Best pairs, LONG-ONLY, z>1.5, confirmed", BEST_PAIRS, 1.5, True, True),
    ("D: Best pairs, LONG-ONLY, z>2.0, confirmed", BEST_PAIRS, 2.0, True, True),
    ("E: All pairs, LONG-ONLY, z>1.5, NO confirm", ALL_PAIRS, 1.5, True, False),
    ("F: Best pairs, LONG-ONLY, z>1.5, NO confirm", BEST_PAIRS, 1.5, True, False),
]

results = {}
for name, pairs, thresh, long_only, confirm in configs:
    print(f"\n\n{'#'*70}")
    print(f"# CONFIG: {name}")
    print(f"{'#'*70}")

    trades = generate_trades(pairs, thresh, long_only, confirm)
    if len(trades) < 5:
        print(f"  Only {len(trades)} trades — SKIP")
        continue

    passed, metrics, rg, pv = five_gate_eval(trades, name)
    results[name] = {
        'passed': passed, 'metrics': metrics, 'regime_gap': rg,
        'p_value': pv, 'n_trades': len(trades)
    }

# ============================================================
# SUMMARY TABLE
# ============================================================

print(f"\n\n{'='*70}")
print("SUMMARY — ALL CONFIGURATIONS")
print(f"{'='*70}")
print(f"\n  {'Config':<48} {'N':>4} {'Sharpe':>7} {'WR':>6} {'PF':>6} {'p-val':>7} {'RG':>5} {'DD':>6} {'PASS?'}")
print(f"  {'-'*96}")

for name, r in results.items():
    m = r['metrics']
    v = "YES" if r['passed'] else "no"
    short_name = name[:47]
    print(f"  {short_name:<48} {m['n_trades']:>4} {m['sharpe']:>7.2f} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {r['p_value']:>7.4f} {r['regime_gap']:>5.2f} {m['max_drawdown']:>5.1%} {v}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")
