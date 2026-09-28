#!/usr/bin/env python3
"""
Adversarial Validation — Sector Re-Rating Acceleration (Variant B)
===================================================================
6-test battery:
  1. Re-implementation (independent code, same logic)
  2. Inverse signal (bottom-3 instead of top-3)
  3. Random timing (1000 shuffled entry dates)
  4. Cost sensitivity (0.05%-0.50%)
  5. Sub-period stability (4 equal periods, all must be Sharpe >= 0)
  6. Parameter robustness (grid search, >=60% combos Sharpe > 0.3)

Signal: Re-Rating Acceleration = 5-day change in (sector_20d_ret - SPY_20d_ret) / sector_20d_vol
        Top-3 sectors by this score, held 5 days, equal-weight.

Author: Claude Opus 4.6 | Date: 2026-08-21
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import sys

# ── Config (matches original Variant B) ──────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

LOOKBACK = 20
ACCEL_DAYS = 5
HOLD_DAYS = 5
TOP_N = 3
COST_RT = 0.0020
ESTIMATION_WINDOW = 60

DATA_START = '2020-01-01'
DATA_END = '2026-08-21'

np.random.seed(42)


# ── Data ─────────────────────────────────────────────────────────────────────
def download_data():
    print("Downloading sector ETF + SPY daily data...", flush=True)
    raw = yf.download(ALL_TICKERS, start=DATA_START, end=DATA_END,
                       progress=False, auto_adjust=True, group_by='ticker', threads=True)
    closes = {}
    for ticker in ALL_TICKERS:
        try:
            if len(ALL_TICKERS) > 1:
                s = raw[ticker]['Close'].dropna().squeeze()
            else:
                s = raw['Close'].dropna().squeeze()
            if len(s) > 100:
                closes[ticker] = s
        except Exception as e:
            print(f"  WARNING: Failed to get {ticker}: {e}", flush=True)
    df = pd.DataFrame(closes).dropna()
    print(f"  Got {len(df)} trading days, {len(df.columns)} tickers", flush=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  ORIGINAL IMPLEMENTATION (from sector_rerate_momentum_backtest.py)
# ══════════════════════════════════════════════════════════════════════════════

def orig_rerate_accel(prices_df, lookback=LOOKBACK, accel_days=ACCEL_DAYS):
    """Original Variant B: 5-day change in re-rating speed."""
    returns = prices_df.pct_change()
    spy_ret = returns[BENCHMARK]

    # Step 1: Compute re-rating speed
    rerate = pd.DataFrame(index=prices_df.index, columns=SECTOR_ETFS, dtype=float)
    for etf in SECTOR_ETFS:
        if etf not in returns.columns:
            continue
        etf_ret = returns[etf]
        etf_cum = etf_ret.rolling(lookback).sum()
        spy_cum = spy_ret.rolling(lookback).sum()
        excess_ret = etf_cum - spy_cum
        etf_vol = etf_ret.rolling(lookback).std() * np.sqrt(252)
        rerate[etf] = excess_ret / etf_vol.replace(0, np.nan)

    # Step 2: Acceleration = 5-day change
    accel = rerate.diff(accel_days)
    return accel


# ══════════════════════════════════════════════════════════════════════════════
#  RE-IMPLEMENTATION (Test 1 — written from scratch)
# ══════════════════════════════════════════════════════════════════════════════

def reimpl_rerate_accel(prices_df, lookback=LOOKBACK, accel_days=ACCEL_DAYS):
    """Independent re-implementation. Uses numpy arrays instead of pandas rolling."""
    n = len(prices_df)
    sector_cols = [c for c in SECTOR_ETFS if c in prices_df.columns]

    # Daily returns as numpy
    px = prices_df.values  # columns in order of prices_df.columns
    col_map = {c: i for i, c in enumerate(prices_df.columns)}
    spy_idx = col_map[BENCHMARK]

    rets = np.diff(px, axis=0) / px[:-1]  # (n-1, ncols)

    result = np.full((n, len(sector_cols)), np.nan)

    for c_idx, etf in enumerate(sector_cols):
        e_idx = col_map[etf]
        etf_r = rets[:, e_idx]
        spy_r = rets[:, spy_idx]

        # Re-rating speed at each point
        rerate_speed = np.full(n, np.nan)

        for i in range(lookback, n - 1):  # rets is (n-1,) so index offset
            # Sum of etf returns over [i-lookback+1 .. i] in rets space
            # which corresponds to prices_df index i+1 (since rets[0] = day1-day0)
            # Actually: rets index i corresponds to price change from day i to day i+1
            # So for day d in prices_df:
            #   lookback-day return ending at day d = sum of rets[d-lookback:d]
            d = i  # in rets space (0-indexed, length n-1)
            if d < lookback:
                continue

            etf_cum = etf_r[d-lookback:d].sum()
            spy_cum = spy_r[d-lookback:d].sum()
            excess = etf_cum - spy_cum
            vol = etf_r[d-lookback:d].std() * np.sqrt(252)

            if vol > 1e-10:
                # Map back to prices_df index: rets[d] is between day d and d+1
                # The lookback ending at rets index d corresponds to prices_df day d+1
                rerate_speed[d + 1] = excess / vol

        # Acceleration: diff over accel_days
        for i in range(accel_days, n):
            if not np.isnan(rerate_speed[i]) and not np.isnan(rerate_speed[i - accel_days]):
                result[i, c_idx] = rerate_speed[i] - rerate_speed[i - accel_days]

    return pd.DataFrame(result, index=prices_df.index, columns=sector_cols)


# ── Backtest Engine ──────────────────────────────────────────────────────────
def run_backtest(prices_df, signal_func, top_n=TOP_N, hold=HOLD_DAYS,
                 cost=COST_RT, invert=False, shuffle_seed=None):
    """
    Walk-forward sector rotation.
    invert=True: pick BOTTOM N instead of TOP N (inverse signal test).
    shuffle_seed: if set, shuffle signal rankings each rebalance.
    """
    signals = signal_func(prices_df)
    returns = prices_df[SECTOR_ETFS].pct_change()
    spy_returns = prices_df[BENCHMARK].pct_change()

    warmup = LOOKBACK + ESTIMATION_WINDOW
    tradeable = prices_df.index[warmup:]

    if len(tradeable) < 40:
        return None

    rng = np.random.RandomState(shuffle_seed) if shuffle_seed is not None else None

    strat_rets = []
    days_since = hold
    holdings = []
    n_rebalances = 0

    for date in tradeable:
        if days_since >= hold:
            day_sig = signals.loc[date].dropna() if date in signals.index else pd.Series(dtype=float)

            if len(day_sig) >= top_n:
                if rng is not None:
                    # Random: shuffle then pick first top_n
                    available = list(day_sig.index)
                    rng.shuffle(available)
                    holdings = available[:top_n]
                elif invert:
                    # Inverse: pick BOTTOM N
                    ranked = day_sig.sort_values(ascending=True)
                    holdings = list(ranked.index[:top_n])
                else:
                    # Normal: pick TOP N
                    ranked = day_sig.sort_values(ascending=False)
                    holdings = list(ranked.index[:top_n])

                n_rebalances += 1
            days_since = 0

        if holdings and date in returns.index:
            dr = returns.loc[date, holdings]
            port_ret = dr.mean()
            if days_since == 0:
                port_ret -= cost / hold
            strat_rets.append(float(port_ret))
        else:
            strat_rets.append(0.0)

        days_since += 1

    sr = pd.Series(strat_rets, index=tradeable[:len(strat_rets)])
    spy_sr = spy_returns.loc[tradeable[:len(strat_rets)]]

    return sr, spy_sr, n_rebalances


def compute_sharpe(rets):
    if len(rets) < 10 or rets.std() == 0:
        return 0.0
    return (rets.mean() / rets.std()) * np.sqrt(252)


def compute_sortino(rets):
    down = rets[rets < 0]
    if len(down) < 5 or down.std() == 0:
        return 0.0
    return (rets.mean() / down.std()) * np.sqrt(252)


def compute_full_metrics(rets, spy_rets):
    sharpe = compute_sharpe(rets)
    sortino = compute_sortino(rets)
    cum = (1 + rets).prod() - 1
    spy_cum = (1 + spy_rets).prod() - 1

    # Max DD
    cum_eq = (1 + rets).cumprod()
    peak = cum_eq.expanding().max()
    dd = (cum_eq - peak) / peak
    max_dd = dd.min()

    # Trade WR (5-day blocks)
    block_rets = []
    for i in range(0, len(rets) - HOLD_DAYS + 1, HOLD_DAYS):
        block_rets.append(rets.iloc[i:i+HOLD_DAYS].sum())
    block_rets = np.array(block_rets)
    trade_wr = (block_rets > 0).mean() if len(block_rets) > 0 else 0
    trade_pf_g = block_rets[block_rets > 0].sum() if len(block_rets) > 0 else 0
    trade_pf_l = abs(block_rets[block_rets < 0].sum()) if len(block_rets) > 0 else 1e-10
    trade_pf = trade_pf_g / max(trade_pf_l, 1e-10)

    # Regime
    green = spy_rets > 0
    red = spy_rets < 0
    sh_g = compute_sharpe(rets[green]) if green.sum() > 10 else 0
    sh_r = compute_sharpe(rets[red]) if red.sum() > 10 else 0
    max_reg = max(abs(sh_g), abs(sh_r))
    regime_gap = abs(sh_g - sh_r) / max_reg if max_reg > 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cum_return_pct': round(cum * 100, 2),
        'spy_return_pct': round(spy_cum * 100, 2),
        'excess_pct': round((cum - spy_cum) * 100, 2),
        'max_dd_pct': round(max_dd * 100, 2),
        'trade_wr': round(trade_wr * 100, 1),
        'trade_pf': round(min(trade_pf, 99), 3),
        'n_trades': len(block_rets),
        'sharpe_green': round(sh_g, 3),
        'sharpe_red': round(sh_r, 3),
        'regime_gap': round(regime_gap, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  6-TEST ADVERSARIAL BATTERY
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 90, flush=True)
    print("ADVERSARIAL VALIDATION — Sector Re-Rating Acceleration (Variant B)", flush=True)
    print("Signal: 5-day change in (excess return / vol) — top-3 sectors, 5-day hold", flush=True)
    print("=" * 90, flush=True)

    prices = download_data()
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n", flush=True)

    scores = {'pass': 0, 'fail': 0, 'tests': {}}

    # ── TEST 1: RE-IMPLEMENTATION ────────────────────────────────────────────
    print("=" * 70, flush=True)
    print("TEST 1: RE-IMPLEMENTATION (independent code)", flush=True)
    print("=" * 70, flush=True)

    orig_result = run_backtest(prices, orig_rerate_accel)
    reimpl_result = run_backtest(prices, reimpl_rerate_accel)

    if orig_result and reimpl_result:
        orig_rets, orig_spy, orig_n = orig_result
        reimpl_rets, reimpl_spy, reimpl_n = reimpl_result

        orig_sharpe = compute_sharpe(orig_rets)
        reimpl_sharpe = compute_sharpe(reimpl_rets)
        orig_m = compute_full_metrics(orig_rets, orig_spy)
        reimpl_m = compute_full_metrics(reimpl_rets, reimpl_spy)

        sharpe_ratio = reimpl_sharpe / orig_sharpe if orig_sharpe != 0 else 0

        print(f"  Original Sharpe:  {orig_sharpe:.3f} (trades: {orig_m['n_trades']})", flush=True)
        print(f"  Re-impl Sharpe:   {reimpl_sharpe:.3f} (trades: {reimpl_m['n_trades']})", flush=True)
        print(f"  Ratio:            {sharpe_ratio:.3f}", flush=True)
        print(f"  Original: WR={orig_m['trade_wr']:.1f}%, PF={orig_m['trade_pf']:.3f}, MDD={orig_m['max_dd_pct']:.2f}%", flush=True)
        print(f"  Re-impl:  WR={reimpl_m['trade_wr']:.1f}%, PF={reimpl_m['trade_pf']:.3f}, MDD={reimpl_m['max_dd_pct']:.2f}%", flush=True)

        # Pass if re-impl Sharpe >= 70% of original (allowing for minor code differences)
        test1_pass = sharpe_ratio >= 0.70 and reimpl_sharpe > 0.5
        print(f"\n  RESULT: {'PASS ✅' if test1_pass else 'FAIL ❌'} (ratio >= 0.70 and reimpl Sharpe > 0.5)", flush=True)
    else:
        test1_pass = False
        print("  RESULT: FAIL ❌ (backtest returned None)", flush=True)

    scores['tests']['re_implementation'] = test1_pass
    scores['pass' if test1_pass else 'fail'] += 1

    # ── TEST 2: INVERSE SIGNAL ───────────────────────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("TEST 2: INVERSE SIGNAL (bottom-3 instead of top-3)", flush=True)
    print("=" * 70, flush=True)

    inv_result = run_backtest(prices, orig_rerate_accel, invert=True)

    if inv_result and orig_result:
        inv_rets, inv_spy, inv_n = inv_result
        inv_sharpe = compute_sharpe(inv_rets)
        inv_m = compute_full_metrics(inv_rets, inv_spy)

        inv_ratio = inv_sharpe / orig_sharpe if orig_sharpe != 0 else 999

        print(f"  Original Sharpe:  {orig_sharpe:.3f}", flush=True)
        print(f"  Inverse Sharpe:   {inv_sharpe:.3f}", flush=True)
        print(f"  Inv/Orig ratio:   {inv_ratio:.3f}", flush=True)
        print(f"  Inverse: WR={inv_m['trade_wr']:.1f}%, PF={inv_m['trade_pf']:.3f}", flush=True)

        # Pass if inverse Sharpe < 0.50 * original (real direction matters)
        test2_pass = inv_ratio < 0.50
        print(f"\n  RESULT: {'PASS ✅' if test2_pass else 'FAIL ❌'} (inverse ratio < 0.50)", flush=True)
    else:
        test2_pass = False
        print("  RESULT: FAIL ❌", flush=True)

    scores['tests']['inverse_signal'] = test2_pass
    scores['pass' if test2_pass else 'fail'] += 1

    # ── TEST 3: RANDOM TIMING (1000 permutations) ────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("TEST 3: RANDOM TIMING (1000 permutations)", flush=True)
    print("=" * 70, flush=True)

    n_perms = 1000
    perm_sharpes = []

    for seed in range(n_perms):
        r = run_backtest(prices, orig_rerate_accel, shuffle_seed=seed + 5000)
        if r:
            perm_sharpes.append(compute_sharpe(r[0]))
        if (seed + 1) % 200 == 0:
            print(f"  ... {seed + 1}/{n_perms} permutations done", flush=True)

    perm_sharpes = np.array(perm_sharpes)
    count_ge = (perm_sharpes >= orig_sharpe).sum()
    perm_p = count_ge / len(perm_sharpes)
    perm_pctl = (perm_sharpes < orig_sharpe).sum() / len(perm_sharpes) * 100

    print(f"  Original Sharpe:     {orig_sharpe:.3f}", flush=True)
    print(f"  Random mean Sharpe:  {perm_sharpes.mean():.3f} ± {perm_sharpes.std():.3f}", flush=True)
    print(f"  Random max Sharpe:   {perm_sharpes.max():.3f}", flush=True)
    print(f"  Percentile:          {perm_pctl:.1f}th", flush=True)
    print(f"  p-value:             {perm_p:.4f}", flush=True)

    test3_pass = perm_p < 0.05
    print(f"\n  RESULT: {'PASS ✅' if test3_pass else 'FAIL ❌'} (p < 0.05)", flush=True)

    scores['tests']['random_timing'] = test3_pass
    scores['pass' if test3_pass else 'fail'] += 1

    # ── TEST 4: COST SENSITIVITY ─────────────────────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("TEST 4: COST SENSITIVITY", flush=True)
    print("=" * 70, flush=True)

    costs = [0.0005, 0.0010, 0.0020, 0.0030, 0.0050]
    cost_sharpes = {}

    for c in costs:
        r = run_backtest(prices, orig_rerate_accel, cost=c)
        if r:
            sh = compute_sharpe(r[0])
            cost_sharpes[c] = sh
            print(f"  Cost {c*100:.2f}%: Sharpe {sh:.3f}", flush=True)

    # Pass if Sharpe > 0.5 at 0.20% (our actual cost) and > 0 at 0.50%
    test4_pass = cost_sharpes.get(0.0020, 0) > 0.5 and cost_sharpes.get(0.0050, 0) > 0
    breakeven = max([c for c, s in cost_sharpes.items() if s > 0], default=0) * 100
    print(f"  Breakeven cost:  ~{breakeven:.2f}%", flush=True)
    print(f"\n  RESULT: {'PASS ✅' if test4_pass else 'FAIL ❌'} (Sharpe > 0.5 at 20bps, > 0 at 50bps)", flush=True)

    scores['tests']['cost_sensitivity'] = test4_pass
    scores['pass' if test4_pass else 'fail'] += 1

    # ── TEST 5: SUB-PERIOD STABILITY ─────────────────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("TEST 5: SUB-PERIOD STABILITY (4 equal periods)", flush=True)
    print("=" * 70, flush=True)

    if orig_result:
        rets = orig_rets
        n = len(rets)
        quarter = n // 4
        sub_sharpes = []

        for q in range(4):
            start = q * quarter
            end = (q + 1) * quarter if q < 3 else n
            sub = rets.iloc[start:end]
            sh = compute_sharpe(sub)
            sub_sharpes.append(sh)
            start_date = sub.index[0].strftime('%Y-%m-%d')
            end_date = sub.index[-1].strftime('%Y-%m-%d')
            print(f"  Q{q+1} ({start_date} to {end_date}): Sharpe {sh:.3f} ({len(sub)} days)", flush=True)

        # Pass if ALL 4 sub-periods have Sharpe >= 0
        test5_pass = all(s >= 0 for s in sub_sharpes)
        improving = sub_sharpes[-1] >= sub_sharpes[0]
        print(f"  Trend: {'IMPROVING' if improving else 'DECLINING'} ({sub_sharpes[0]:.3f} → {sub_sharpes[-1]:.3f})", flush=True)
        print(f"\n  RESULT: {'PASS ✅' if test5_pass else 'FAIL ❌'} (all sub-periods Sharpe >= 0)", flush=True)
    else:
        test5_pass = False
        print("  RESULT: FAIL ❌", flush=True)

    scores['tests']['sub_period'] = test5_pass
    scores['pass' if test5_pass else 'fail'] += 1

    # ── TEST 6: PARAMETER ROBUSTNESS ─────────────────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("TEST 6: PARAMETER ROBUSTNESS (grid search)", flush=True)
    print("=" * 70, flush=True)

    lookbacks = [10, 15, 20, 25, 30]
    accel_days_grid = [3, 5, 7, 10]
    top_ns = [2, 3, 4]
    holds = [3, 5, 7, 10]

    total_combos = len(lookbacks) * len(accel_days_grid) * len(top_ns) * len(holds)
    print(f"  Testing {total_combos} parameter combinations...", flush=True)

    param_sharpes = []
    good_count = 0
    tested = 0

    for lb in lookbacks:
        for ad in accel_days_grid:
            for tn in top_ns:
                for hd in holds:
                    sig_func = lambda df, _lb=lb, _ad=ad: orig_rerate_accel(df, _lb, _ad)
                    r = run_backtest(prices, sig_func, top_n=tn, hold=hd)
                    if r:
                        sh = compute_sharpe(r[0])
                        param_sharpes.append(sh)
                        if sh > 0.3:
                            good_count += 1
                    tested += 1

        print(f"  ... {tested}/{total_combos} tested", flush=True)

    param_sharpes = np.array(param_sharpes)
    pct_good = good_count / len(param_sharpes) * 100 if len(param_sharpes) > 0 else 0

    print(f"  Total combos tested: {len(param_sharpes)}", flush=True)
    print(f"  Combos with Sharpe > 0.3: {good_count} ({pct_good:.1f}%)", flush=True)
    print(f"  Sharpe distribution: mean={param_sharpes.mean():.3f}, median={np.median(param_sharpes):.3f}, "
          f"std={param_sharpes.std():.3f}", flush=True)
    print(f"  Range: [{param_sharpes.min():.3f}, {param_sharpes.max():.3f}]", flush=True)

    # Pass if >= 60% of combos have Sharpe > 0.3
    test6_pass = pct_good >= 60
    print(f"\n  RESULT: {'PASS ✅' if test6_pass else 'FAIL ❌'} ({pct_good:.1f}% >= 60% threshold)", flush=True)

    scores['tests']['parameter_robustness'] = test6_pass
    scores['pass' if test6_pass else 'fail'] += 1

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 90}", flush=True)
    print("ADVERSARIAL VALIDATION SUMMARY — Sector Re-Rating Acceleration", flush=True)
    print(f"{'=' * 90}", flush=True)

    for test_name, passed in scores['tests'].items():
        status = "PASS ✅" if passed else "FAIL ❌"
        print(f"  {test_name:<25s}: {status}", flush=True)

    total_pass = scores['pass']
    total = scores['pass'] + scores['fail']
    print(f"\n  TOTAL: {total_pass}/{total} tests passed", flush=True)

    if total_pass == 6:
        print(f"\n  🏆🏆🏆 PERFECT ADVERSARIAL PASS — STRATEGY VALIDATED", flush=True)
    elif total_pass >= 5:
        print(f"\n  🏆🏆 STRONG PASS ({total_pass}/6) — STRATEGY VALIDATED WITH MINOR CONCERNS", flush=True)
    elif total_pass >= 4:
        print(f"\n  🟡 PARTIAL PASS ({total_pass}/6) — NEEDS FURTHER INVESTIGATION", flush=True)
    else:
        print(f"\n  ❌ FAILED ({total_pass}/6) — STRATEGY NOT VALIDATED", flush=True)

    # Key metrics
    if orig_result:
        m = compute_full_metrics(orig_rets, orig_spy)
        print(f"\n  Key Metrics:", flush=True)
        print(f"    Sharpe:        {m['sharpe']:.3f}", flush=True)
        print(f"    Sortino:       {m['sortino']:.3f}", flush=True)
        print(f"    Trade WR:      {m['trade_wr']:.1f}%", flush=True)
        print(f"    Trade PF:      {m['trade_pf']:.3f}", flush=True)
        print(f"    Max DD:        {m['max_dd_pct']:.2f}%", flush=True)
        print(f"    Excess vs SPY: {m['excess_pct']:.2f}%", flush=True)
        print(f"    Regime gap:    {m['regime_gap']:.4f}", flush=True)
        print(f"    Green Sharpe:  {m['sharpe_green']:.3f}", flush=True)
        print(f"    Red Sharpe:    {m['sharpe_red']:.3f}", flush=True)

    print(f"\n  Adversarial details:", flush=True)
    if orig_result and reimpl_result:
        print(f"    Re-impl ratio: {reimpl_sharpe/orig_sharpe:.3f}", flush=True)
    if inv_result:
        print(f"    Inverse Sharpe: {inv_sharpe:.3f} (ratio: {inv_ratio:.3f})", flush=True)
    print(f"    Random p-value: {perm_p:.4f}", flush=True)
    print(f"    Breakeven cost: ~{breakeven:.2f}%", flush=True)
    print(f"    Sub-period Sharpes: {[round(s,3) for s in sub_sharpes]}", flush=True)
    print(f"    Param robustness: {pct_good:.1f}%", flush=True)

    print(f"\nDone. {datetime.now()}", flush=True)


if __name__ == '__main__':
    main()
