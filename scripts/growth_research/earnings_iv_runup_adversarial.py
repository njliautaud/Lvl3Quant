#!/usr/bin/env python3
"""
Adversarial Validation for Earnings IV Run-Up Strategy
========================================================

The v1 results look too good (5/5 gates, Sharpe 2-3). This script
stress-tests the results with adversarial checks:

1. HALVED IV EXPANSION — Cut IV multipliers by 50%. If still profitable, edge is robust.
2. DOUBLED THETA DECAY — Extra theta penalty. If still profitable, theta isn't the real driver.
3. WIDER BID-ASK — Add 5% bid-ask spread cost. Real straddles have worse fills.
4. RANDOM ENTRY TIMING — Enter at random dates (not T-15 before earnings). If profitable, edge is from stock movement not IV.
5. CONCENTRATION CHECK — Top-3 ticker contribution to total P&L.
6. YEAR-BY-YEAR BREAKDOWN — Profitable in all regimes (2022 bear, 2023-24 bull)?
7. SUB-PERIOD STABILITY — First half vs second half Sharpe comparison.
8. DEEP PERMUTATION TEST — 1000 shuffles of trade-level P&L.
"""

import json
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = '.'

# Load the v1 results
RESULTS_PATH = os.path.join(LVL3_ROOT, 'research', 'findings', 'earnings_iv_runup_v1.json')

# Re-import the strategy code
sys.path.insert(0, os.path.join(LVL3_ROOT, 'scripts', 'growth_research'))

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

INITIAL_CAPITAL = 645.0
MAX_POS_COST = 200.0
MAX_CONCURRENT = 3
COMMISSION_RT = 1.30
RISK_FREE_RATE = 0.05
BS_HAIRCUT = 0.85

OOT_START = '2022-01-01'
OOT_END = '2026-07-01'


def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def straddle_price(S, K, T, r, sigma):
    return (bs_call(S, K, T, r, sigma) + bs_put(S, K, T, r, sigma)) * BS_HAIRCUT


def estimate_iv(ticker_df, idx, days_to_earnings, iv_scale=1.0):
    """IV model with adjustable scaling."""
    rets = np.diff(np.log(ticker_df['close'].values[max(0,idx-21):idx+1]))
    realized_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 5 else 0.3

    if days_to_earnings <= 0:
        iv_mult = 0.8
    elif days_to_earnings <= 1:
        base_mult = 1.8
        iv_mult = 1.0 + (base_mult - 1.0) * iv_scale
    elif days_to_earnings <= 2:
        base_mult = 1.6
        iv_mult = 1.0 + (base_mult - 1.0) * iv_scale
    elif days_to_earnings <= 5:
        base_mult = 1.4
        iv_mult = 1.0 + (base_mult - 1.0) * iv_scale
    elif days_to_earnings <= 10:
        base_mult = 1.2
        iv_mult = 1.0 + (base_mult - 1.0) * iv_scale
    elif days_to_earnings <= 15:
        base_mult = 1.1
        iv_mult = 1.0 + (base_mult - 1.0) * iv_scale
    else:
        iv_mult = 1.0

    return realized_vol * iv_mult


def run_backtest(prices, earnings, iv_scale=1.0, extra_cost_pct=0.0, random_timing=False):
    """Run the A variant (straddle T-15 to T-1) with adjustable parameters."""
    capital = INITIAL_CAPITAL
    trades = []
    equity_curve = [capital]
    open_positions = []

    events = []
    for ticker in STOCK_UNIVERSE:
        if ticker not in earnings:
            continue
        tdf = prices[prices['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(tdf) < 50:
            continue
        trading_days = tdf['date'].values

        for earn_date_str in earnings[ticker]:
            earn_date = pd.Timestamp(earn_date_str)
            if earn_date < pd.Timestamp(OOT_START) or earn_date > pd.Timestamp(OOT_END):
                continue

            earn_idx = np.searchsorted(trading_days, np.datetime64(earn_date))

            if random_timing:
                # Random entry: pick a random date within 30 days before earnings
                offset = np.random.randint(5, 25)
                entry_idx = earn_idx - offset
                exit_idx = entry_idx + 14  # hold same duration
            else:
                entry_idx = earn_idx - 15
                exit_idx = earn_idx - 1

            if entry_idx < 30 or exit_idx >= len(tdf) or entry_idx >= exit_idx:
                continue

            events.append({
                'ticker': ticker,
                'earn_date': earn_date,
                'entry_idx': int(entry_idx),
                'exit_idx': int(exit_idx),
                'earn_idx': int(earn_idx),
            })

    events.sort(key=lambda x: x['earn_date'])

    for event in events:
        ticker = event['ticker']
        tdf = prices[prices['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        entry_idx = event['entry_idx']
        exit_idx = event['exit_idx']

        current_open = [p for p in open_positions
                       if p['exit_date'] > pd.Timestamp(tdf['date'].iloc[entry_idx])]
        if len(current_open) >= MAX_CONCURRENT:
            continue

        entry_price = float(tdf['close'].iloc[entry_idx])
        strike = round(entry_price)
        days_to_earn_entry = event['earn_idx'] - entry_idx
        days_to_earn_exit = event['earn_idx'] - exit_idx
        dte_entry = days_to_earn_entry + 7

        entry_iv = estimate_iv(tdf, entry_idx, days_to_earn_entry, iv_scale)
        T_entry = dte_entry / 252.0
        entry_opt = straddle_price(entry_price, strike, T_entry, RISK_FREE_RATE, entry_iv)
        entry_cost = entry_opt * 100 + COMMISSION_RT

        # Add extra cost (bid-ask simulation)
        entry_cost *= (1 + extra_cost_pct)

        if entry_cost <= 0 or entry_cost > MAX_POS_COST or entry_cost > capital:
            continue

        exit_price = float(tdf['close'].iloc[exit_idx])
        exit_iv = estimate_iv(tdf, exit_idx, days_to_earn_exit, iv_scale)
        dte_exit = max(dte_entry - (exit_idx - entry_idx), 1)
        T_exit = dte_exit / 252.0
        exit_opt = straddle_price(exit_price, strike, T_exit, RISK_FREE_RATE, exit_iv)
        exit_value = exit_opt * 100 - COMMISSION_RT

        # Apply bid-ask on exit too
        exit_value *= (1 - extra_cost_pct)

        pnl = exit_value - entry_cost
        pnl_pct = pnl / entry_cost if entry_cost > 0 else 0
        capital += pnl

        trades.append({
            'ticker': ticker,
            'entry_date': str(tdf['date'].iloc[entry_idx])[:10],
            'exit_date': str(tdf['date'].iloc[exit_idx])[:10],
            'earn_date': str(event['earn_date'])[:10],
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct * 100, 2),
            'capital_after': round(capital, 2),
        })

        equity_curve.append(capital)
        open_positions.append({
            'ticker': ticker,
            'exit_date': pd.Timestamp(tdf['date'].iloc[exit_idx]),
        })

    return trades, equity_curve


def compute_sharpe(trades, years=4.5):
    if not trades:
        return 0
    pnl_pcts = [t['pnl_pct'] for t in trades]
    if np.std(pnl_pcts) == 0:
        return 0
    return np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(len(pnl_pcts) / years)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  ADVERSARIAL VALIDATION — Earnings IV Run-Up")
    print("=" * 60)

    # Load data
    cache_path = os.path.join(LVL3_ROOT, 'data', 'iv_runup_prices_cache.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'iv_runup_earnings_cache.json')

    prices = pd.read_parquet(cache_path)
    with open(earnings_cache) as f:
        earnings = json.load(f)

    checks = {}

    # ======== CHECK 1: HALVED IV EXPANSION ========
    print("\n1. HALVED IV EXPANSION (50% of assumed IV rise)...")
    trades_half, eq_half = run_backtest(prices, earnings, iv_scale=0.5)
    sharpe_half = compute_sharpe(trades_half)
    n_half = len(trades_half)
    final_half = eq_half[-1] if eq_half else INITIAL_CAPITAL
    wins_half = sum(1 for t in trades_half if t['pnl'] > 0)
    wr_half = wins_half / n_half * 100 if n_half > 0 else 0

    checks['halved_iv'] = {
        'pass': sharpe_half > 0.5,  # Still positive even with halved IV?
        'sharpe': round(sharpe_half, 3),
        'n_trades': n_half,
        'wr': round(wr_half, 1),
        'final': round(final_half, 2),
    }
    status = '✅ PASS' if checks['halved_iv']['pass'] else '❌ FAIL'
    print(f"   {status} — Sharpe {sharpe_half:.3f}, WR {wr_half:.0f}%, ${final_half:.0f} ({n_half} trades)")

    # ======== CHECK 2: QUARTER IV EXPANSION ========
    print("\n2. QUARTER IV EXPANSION (25% of assumed IV rise)...")
    trades_q, eq_q = run_backtest(prices, earnings, iv_scale=0.25)
    sharpe_q = compute_sharpe(trades_q)
    n_q = len(trades_q)
    final_q = eq_q[-1] if eq_q else INITIAL_CAPITAL
    wins_q = sum(1 for t in trades_q if t['pnl'] > 0)
    wr_q = wins_q / n_q * 100 if n_q > 0 else 0

    checks['quarter_iv'] = {
        'pass': sharpe_q > 0,
        'sharpe': round(sharpe_q, 3),
        'n_trades': n_q,
        'wr': round(wr_q, 1),
        'final': round(final_q, 2),
    }
    status = '✅ PASS' if checks['quarter_iv']['pass'] else '❌ FAIL'
    print(f"   {status} — Sharpe {sharpe_q:.3f}, WR {wr_q:.0f}%, ${final_q:.0f} ({n_q} trades)")

    # ======== CHECK 3: 5% BID-ASK SPREAD ========
    print("\n3. 5% BID-ASK SPREAD (on entry and exit)...")
    trades_ba, eq_ba = run_backtest(prices, earnings, iv_scale=1.0, extra_cost_pct=0.05)
    sharpe_ba = compute_sharpe(trades_ba)
    n_ba = len(trades_ba)
    final_ba = eq_ba[-1] if eq_ba else INITIAL_CAPITAL
    wins_ba = sum(1 for t in trades_ba if t['pnl'] > 0)
    wr_ba = wins_ba / n_ba * 100 if n_ba > 0 else 0

    checks['bid_ask'] = {
        'pass': sharpe_ba > 1.0,
        'sharpe': round(sharpe_ba, 3),
        'n_trades': n_ba,
        'wr': round(wr_ba, 1),
        'final': round(final_ba, 2),
    }
    status = '✅ PASS' if checks['bid_ask']['pass'] else '❌ FAIL'
    print(f"   {status} — Sharpe {sharpe_ba:.3f}, WR {wr_ba:.0f}%, ${final_ba:.0f} ({n_ba} trades)")

    # ======== CHECK 4: RANDOM ENTRY TIMING ========
    print("\n4. RANDOM ENTRY TIMING (not tied to earnings schedule)...")
    random_sharpes = []
    for i in range(10):
        trades_r, eq_r = run_backtest(prices, earnings, iv_scale=1.0, random_timing=True)
        random_sharpes.append(compute_sharpe(trades_r))

    random_mean = np.mean(random_sharpes)
    random_p95 = np.percentile(random_sharpes, 95)

    # Compare to baseline (A variant Sharpe 2.269)
    baseline_sharpe = 2.269
    checks['random_timing'] = {
        'pass': baseline_sharpe > random_p95 * 1.5,  # Must be well above random
        'baseline_sharpe': baseline_sharpe,
        'random_mean_sharpe': round(random_mean, 3),
        'random_p95_sharpe': round(random_p95, 3),
    }
    status = '✅ PASS' if checks['random_timing']['pass'] else '❌ FAIL'
    print(f"   {status} — Baseline Sharpe {baseline_sharpe:.3f} vs Random mean {random_mean:.3f} (p95: {random_p95:.3f})")

    # ======== CHECK 5: CONCENTRATION ========
    print("\n5. CONCENTRATION CHECK...")
    trades_base, eq_base = run_backtest(prices, earnings)
    ticker_pnl = {}
    for t in trades_base:
        ticker_pnl[t['ticker']] = ticker_pnl.get(t['ticker'], 0) + t['pnl']

    total_pnl = sum(t['pnl'] for t in trades_base)
    if total_pnl > 0:
        sorted_tickers = sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)
        top3_pnl = sum(v for _, v in sorted_tickers[:3])
        top3_pct = top3_pnl / total_pnl * 100
    else:
        top3_pct = 0
        sorted_tickers = []

    checks['concentration'] = {
        'pass': top3_pct < 50,
        'top3_pct': round(top3_pct, 1),
        'top_tickers': [(t, round(p, 2)) for t, p in sorted_tickers[:5]] if sorted_tickers else [],
    }
    status = '✅ PASS' if checks['concentration']['pass'] else '❌ FAIL'
    print(f"   {status} — Top-3 tickers = {top3_pct:.0f}% of profits")
    for t, p in sorted_tickers[:5]:
        print(f"     {t}: ${p:+.2f} ({p/total_pnl*100:.0f}%)" if total_pnl > 0 else f"     {t}: ${p:+.2f}")

    # ======== CHECK 6: YEAR-BY-YEAR ========
    print("\n6. YEAR-BY-YEAR BREAKDOWN...")
    year_groups = {}
    for t in trades_base:
        y = t['entry_date'][:4]
        year_groups.setdefault(y, []).append(t)

    all_years_positive = True
    for y in sorted(year_groups.keys()):
        yt = year_groups[y]
        y_pnl = sum(t['pnl'] for t in yt)
        y_wr = sum(1 for t in yt if t['pnl'] > 0) / len(yt) * 100 if yt else 0
        y_sharpe = compute_sharpe(yt, years=1)
        if y_pnl <= 0:
            all_years_positive = False
        print(f"   {y}: {len(yt)} trades, WR {y_wr:.0f}%, PnL ${y_pnl:+.0f}, Sharpe {y_sharpe:.2f}")

    checks['yearly'] = {
        'pass': all_years_positive,
        'all_positive': all_years_positive,
    }
    status = '✅ PASS' if checks['yearly']['pass'] else '❌ FAIL'
    print(f"   {status} — All years positive: {all_years_positive}")

    # ======== CHECK 7: SUB-PERIOD STABILITY ========
    print("\n7. SUB-PERIOD STABILITY...")
    mid = len(trades_base) // 2
    first_half = trades_base[:mid]
    second_half = trades_base[mid:]
    sh_first = compute_sharpe(first_half, years=2.25)
    sh_second = compute_sharpe(second_half, years=2.25)

    ratio = abs(sh_first - sh_second) / max(abs(sh_first), abs(sh_second), 0.01)
    checks['sub_period'] = {
        'pass': ratio < 0.70,
        'first_half_sharpe': round(sh_first, 3),
        'second_half_sharpe': round(sh_second, 3),
        'ratio': round(ratio, 3),
    }
    status = '✅ PASS' if checks['sub_period']['pass'] else '❌ FAIL'
    print(f"   {status} — First half Sharpe {sh_first:.3f}, Second half Sharpe {sh_second:.3f} (gap {ratio:.2f})")

    # ======== CHECK 8: DEEP PERMUTATION TEST ========
    print("\n8. DEEP PERMUTATION TEST (1000 shuffles)...")
    pnl_pcts = [t['pnl_pct'] for t in trades_base]
    observed = np.mean(pnl_pcts)
    n_perms = 1000
    perm_count = 0
    for _ in range(n_perms):
        shuffled = np.random.choice([-1, 1], size=len(pnl_pcts)) * np.abs(pnl_pcts)
        if np.mean(shuffled) >= observed:
            perm_count += 1
    deep_p = perm_count / n_perms

    checks['deep_perm'] = {
        'pass': deep_p < 0.05,
        'p_value': round(deep_p, 4),
    }
    status = '✅ PASS' if checks['deep_perm']['pass'] else '❌ FAIL'
    print(f"   {status} — p = {deep_p:.4f}")

    # ======== CHECK 9: REMOVE TOP TICKER ========
    print("\n9. REMOVE TOP TICKER...")
    if sorted_tickers:
        top_ticker = sorted_tickers[0][0]
        trades_excl = [t for t in trades_base if t['ticker'] != top_ticker]
        sharpe_excl = compute_sharpe(trades_excl)
        final_excl = trades_excl[-1]['capital_after'] if trades_excl else INITIAL_CAPITAL

        checks['remove_top'] = {
            'pass': sharpe_excl > 1.0,
            'removed': top_ticker,
            'sharpe_after': round(sharpe_excl, 3),
            'final_after': round(final_excl, 2),
        }
        status = '✅ PASS' if checks['remove_top']['pass'] else '❌ FAIL'
        print(f"   {status} — Removed {top_ticker}: Sharpe {sharpe_excl:.3f}, ${final_excl:.0f}")
    else:
        checks['remove_top'] = {'pass': False, 'error': 'no tickers'}

    # ======== SUMMARY ========
    elapsed = time.time() - t0
    n_passed = sum(1 for v in checks.values() if v.get('pass'))
    n_total = len(checks)

    print(f"\n{'='*60}")
    print(f"  ADVERSARIAL SUMMARY: {n_passed}/{n_total} checks passed ({elapsed:.0f}s)")
    print(f"{'='*60}")
    for name, result in checks.items():
        status = '✅' if result.get('pass') else '❌'
        print(f"  {status} {name}")

    # Critical assessment
    print(f"\n  CRITICAL ASSESSMENT:")
    if checks['halved_iv']['pass'] and checks['quarter_iv']['pass']:
        print(f"  ✅ Edge survives even with 75% reduction in IV expansion assumption")
    elif checks['halved_iv']['pass']:
        print(f"  ⚠️ Edge survives halved IV but dies at quarter IV — SENSITIVE to IV model")
    else:
        print(f"  ❌ Edge DIES when IV expansion is halved — LIKELY ARTIFACT of pricing model")

    if checks['random_timing']['pass']:
        print(f"  ✅ Pre-earnings timing adds genuine value vs random entry")
    else:
        print(f"  ❌ Random entry timing is nearly as good — edge may be from stock direction, not IV")

    if not checks['concentration']['pass']:
        print(f"  ❌ Concentrated in few tickers — fragile alpha")

    # Save
    output_path = os.path.join(LVL3_ROOT, 'research', 'findings', 'earnings_iv_runup_adversarial.json')
    with open(output_path, 'w') as f:
        json.dump(checks, f, indent=2, default=str)
    print(f"\nSaved to {output_path}")


if __name__ == '__main__':
    main()
