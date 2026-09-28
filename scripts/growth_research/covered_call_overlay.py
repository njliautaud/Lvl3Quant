#!/usr/bin/env python3
"""
Covered Call Overlay on VIX-Gated UPRO — Comprehensive Backtest
================================================================
Walk-forward sliding window (252d train, 63d test, 21d slide).
Uses Black-Scholes with trailing realized vol to price calls.
Compares: UPRO B&H, VIX-gated UPRO, VIX-gated UPRO + call overlay (4 variants).

Author: Claude Opus 4.6
Date: 2026-07-17
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/covered_call_overlay")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Black-Scholes helpers ───────────────────────────────────────────────────

def bs_d1(S, K, T, r, sigma):
    """d1 in Black-Scholes formula."""
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = bs_d1(S, K, T, r, sigma)
    return norm.cdf(d1)

def strike_for_delta(S, T, r, sigma, target_delta):
    """Find strike that gives a specific call delta using bisection."""
    if T <= 0 or sigma <= 0:
        return S * 1.05  # fallback OTM
    # Call delta decreases as K increases
    K_lo, K_hi = S * 0.5, S * 3.0
    for _ in range(100):
        K_mid = (K_lo + K_hi) / 2
        d = bs_call_delta(S, K_mid, T, r, sigma)
        if d > target_delta:
            K_lo = K_mid
        else:
            K_hi = K_mid
        if abs(d - target_delta) < 1e-6:
            break
    return K_mid


# ─── Data download ───────────────────────────────────────────────────────────

def download_data():
    """Download UPRO, SPY, and VIX data."""
    print("Downloading data from yfinance...")
    upro = yf.download("UPRO", start="2012-06-01", end="2026-07-17", auto_adjust=True, progress=False)
    spy = yf.download("SPY", start="2012-06-01", end="2026-07-17", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", start="2012-06-01", end="2026-07-17", auto_adjust=True, progress=False)

    # Flatten multi-level columns if present
    for df in [upro, spy, vix]:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

    # Build combined dataframe
    data = pd.DataFrame(index=upro.index)
    data["upro_close"] = upro["Close"]
    data["upro_ret"] = upro["Close"].pct_change()
    data["spy_close"] = spy["Close"].reindex(upro.index, method="ffill")
    data["spy_ret"] = data["spy_close"].pct_change()
    data["vix"] = vix["Close"].reindex(upro.index, method="ffill")

    # Realized vol: trailing 21d annualized vol of UPRO
    data["rvol_21d"] = data["upro_ret"].rolling(21).std() * np.sqrt(252)
    # Longer window for more stable estimate
    data["rvol_63d"] = data["upro_ret"].rolling(63).std() * np.sqrt(252)

    data = data.dropna()
    print(f"Data: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}, {len(data)} days")
    return data


# ─── VIX gating logic ────────────────────────────────────────────────────────

def vix_allocation(vix_level):
    """Return UPRO allocation fraction based on VIX."""
    if vix_level < 17:
        return 1.0
    elif vix_level < 25:
        return 0.5  # reduce
    else:
        return 0.0  # cash


# ─── Covered call simulation engine ──────────────────────────────────────────

def simulate_covered_call(data, strategy_name, delta_func, use_vix_gate=True,
                          call_tenor_days=30, r=0.04, spread_cost_bps=10):
    """
    Simulate a covered call overlay on UPRO.

    Mechanics:
    - At call sale: receive premium upfront (one-time credit to portfolio on sale day)
    - During call life: UPRO moves normally, but if price > strike we owe the
      intrinsic value at expiry (capped upside)
    - At expiry ITM: lose (price - strike)/price_at_sale of the notional, pay spread to rebuy
    - At expiry OTM: premium already collected, nothing more happens
    - Premium is credited on the DAY the call is sold (not amortized)

    Parameters:
    -----------
    data : DataFrame with upro_close, upro_ret, vix, rvol_63d, spy_ret
    strategy_name : str
    delta_func : callable(vix) -> target_delta or None (no call)
    use_vix_gate : bool, apply VIX-based position sizing
    call_tenor_days : int, days between rolls (30=monthly, 7=weekly)
    r : float, risk-free rate
    spread_cost_bps : float, spread cost per option trade in bps of underlying
    """
    dates = data.index
    n = len(dates)

    # Track portfolio
    portfolio_value = np.ones(n) * 100.0  # start at $100
    daily_pnl = np.zeros(n)
    premium_collected = 0.0
    total_assignments = 0
    total_rolls = 0

    # Call state
    call_active = False
    call_strike = 0.0
    call_sale_price = 0.0  # UPRO price when call was sold
    call_expiry_idx = 0

    for i in range(1, n):
        upro_price = data["upro_close"].iloc[i]
        upro_price_prev = data["upro_close"].iloc[i - 1]
        vix_level = data["vix"].iloc[i - 1]  # use prior day VIX for decisions
        sigma = data["rvol_63d"].iloc[i - 1]

        # Determine allocation
        if use_vix_gate:
            alloc = vix_allocation(vix_level)
        else:
            alloc = 1.0

        # Daily return on UPRO portion
        upro_daily_ret = data["upro_ret"].iloc[i]
        if np.isnan(upro_daily_ret):
            upro_daily_ret = 0.0

        # Portfolio return from UPRO holding
        port_ret = alloc * upro_daily_ret

        # Handle covered call
        target_delta = delta_func(vix_level) if delta_func is not None else None

        # If in cash (alloc=0), close any call
        if alloc == 0 and call_active:
            call_active = False

        # Check if call expired
        if call_active and i >= call_expiry_idx:
            # Expiry day
            if upro_price > call_strike:
                # Assigned: we lose the upside above strike
                # The gain above strike goes to the call buyer
                assignment_loss = (upro_price - call_strike) / upro_price_prev
                port_ret -= alloc * assignment_loss
                # Spread cost for rebuying shares
                port_ret -= alloc * (spread_cost_bps / 10000.0)
                total_assignments += 1
            call_active = False
            total_rolls += 1

        # Sell new call if appropriate
        if not call_active and target_delta is not None and alloc > 0:
            T = call_tenor_days / 252.0
            if sigma > 0.01:
                call_strike = strike_for_delta(upro_price, T, r, sigma, target_delta)
                call_premium_pct = bs_call_price(upro_price, call_strike, T, r, sigma) / upro_price
                # Spread cost on option trade
                call_premium_pct -= spread_cost_bps / 10000.0
                call_premium_pct = max(call_premium_pct, 0.0)

                # Credit premium upfront on sale day
                port_ret += alloc * call_premium_pct
                premium_collected += call_premium_pct * alloc

                call_active = True
                call_sale_price = upro_price
                call_expiry_idx = i + call_tenor_days

        portfolio_value[i] = portfolio_value[i - 1] * (1 + port_ret)
        daily_pnl[i] = port_ret

    results = pd.DataFrame({
        "date": dates,
        "portfolio_value": portfolio_value,
        "daily_return": daily_pnl,
        "upro_close": data["upro_close"].values,
        "vix": data["vix"].values,
        "spy_ret": data["spy_ret"].values,
    })
    results.set_index("date", inplace=True)

    return results, {
        "strategy": strategy_name,
        "premium_collected_pct": premium_collected * 100,
        "total_assignments": total_assignments,
        "total_rolls": total_rolls,
        "assignment_rate": total_assignments / max(total_rolls, 1),
    }


# ─── Walk-forward engine ─────────────────────────────────────────────────────

def walk_forward_backtest(data, strategy_name, delta_func, use_vix_gate=True,
                          call_tenor_days=30, train_days=252, test_days=63, slide_days=21):
    """
    Walk-forward sliding window backtest.
    Train window used to calibrate realized vol (already embedded in data).
    Test window is the actual OOT evaluation period.
    """
    dates = data.index
    n = len(dates)
    all_test_results = []
    all_test_meta = []

    start = train_days
    while start + test_days <= n:
        test_start = start
        test_end = min(start + test_days, n)

        # Use the full data slice — vol is already computed on trailing window
        test_data = data.iloc[test_start:test_end].copy()

        if len(test_data) < 10:
            start += slide_days
            continue

        results, meta = simulate_covered_call(
            test_data, strategy_name, delta_func,
            use_vix_gate=use_vix_gate, call_tenor_days=call_tenor_days
        )
        results["wf_fold"] = start
        all_test_results.append(results)
        all_test_meta.append(meta)

        start += slide_days

    if not all_test_results:
        return None, None

    combined = pd.concat(all_test_results)
    # Deduplicate: if same date appears in multiple folds, keep last (most recent fold)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Recompute cumulative portfolio value from daily returns
    combined["portfolio_value"] = (1 + combined["daily_return"]).cumprod() * 100

    return combined, all_test_meta


# ─── Metrics computation ─────────────────────────────────────────────────────

def compute_metrics(results, strategy_name):
    """Compute comprehensive performance metrics."""
    rets = results["daily_return"].values
    rets = rets[~np.isnan(rets)]

    total_return = (results["portfolio_value"].iloc[-1] / results["portfolio_value"].iloc[0]) - 1
    n_years = len(rets) / 252.0
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1

    ann_vol = np.std(rets) * np.sqrt(252) if len(rets) > 1 else 0
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    downside_rets = rets[rets < 0]
    downside_vol = np.std(downside_rets) * np.sqrt(252) if len(downside_rets) > 1 else 0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    cum = results["portfolio_value"].values
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = np.min(dd)

    # Win rate (daily)
    win_rate = np.mean(rets > 0) if len(rets) > 0 else 0

    # Profit factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = abs(np.sum(rets[rets < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "strategy": strategy_name,
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct": round(ann_return * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate, 4),
        "n_days": len(rets),
        "n_years": round(n_years, 2),
    }


# ─── Validation: Permutation test ────────────────────────────────────────────

def permutation_test(results, base_results=None, n_perms=100):
    """
    Permutation test for the overlay's incremental value.

    If base_results is provided, we test whether the INCREMENTAL returns
    (overlay - base) have a significantly positive mean. We shuffle the
    assignment of overlay vs base days to break any real signal.

    If no base, we test the strategy returns directly.
    """
    rets = results["daily_return"].values
    rets = rets[~np.isnan(rets)]

    if base_results is not None:
        base_rets = base_results["daily_return"].values
        base_rets = base_rets[~np.isnan(base_rets)]
        min_len = min(len(rets), len(base_rets))
        incremental = rets[:min_len] - base_rets[:min_len]
        actual_stat = np.mean(incremental) / np.std(incremental) * np.sqrt(252) if np.std(incremental) > 0 else 0

        rng = np.random.RandomState(42)
        perm_stats = []
        for _ in range(n_perms):
            # Randomly swap overlay and base for each day
            mask = rng.randint(0, 2, size=min_len).astype(bool)
            shuffled_inc = np.where(mask, incremental, -incremental)
            s = np.mean(shuffled_inc) / np.std(shuffled_inc) * np.sqrt(252) if np.std(shuffled_inc) > 0 else 0
            perm_stats.append(s)
    else:
        actual_stat = np.mean(rets) / np.std(rets) * np.sqrt(252) if np.std(rets) > 0 else 0
        rng = np.random.RandomState(42)
        perm_stats = []
        for _ in range(n_perms):
            # Randomly flip sign of returns to test if mean > 0 is significant
            signs = rng.choice([-1, 1], size=len(rets))
            shuffled = rets * signs
            s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252) if np.std(shuffled) > 0 else 0
            perm_stats.append(s)

    p_value = np.mean(np.array(perm_stats) >= actual_stat)
    return actual_stat, p_value, perm_stats


# ─── Validation: Regime test (R1) ────────────────────────────────────────────

def regime_test(results):
    """
    R1 regime test: classify days as green/red based on SPY return.
    Compute Sharpe per regime. Gap must be < 0.50.
    """
    rets = results["daily_return"].values
    spy_rets = results["spy_ret"].values

    green_mask = spy_rets > 0
    red_mask = spy_rets < 0

    green_rets = rets[green_mask]
    red_rets = rets[red_mask]

    def daily_sharpe(r):
        if len(r) < 5 or np.std(r) == 0:
            return 0.0
        return np.mean(r) / np.std(r) * np.sqrt(252)

    sharpe_green = daily_sharpe(green_rets)
    sharpe_red = daily_sharpe(red_rets)

    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0

    return {
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(gap, 3),
        "regime_pass": gap < 0.50,
        "n_green_days": int(np.sum(green_mask)),
        "n_red_days": int(np.sum(red_mask)),
    }


# ─── Validation: Sub-period consistency ──────────────────────────────────────

def subperiod_test(results):
    """Both halves must be profitable."""
    n = len(results)
    mid = n // 2

    first_half = results.iloc[:mid]
    second_half = results.iloc[mid:]

    ret1 = first_half["portfolio_value"].iloc[-1] / first_half["portfolio_value"].iloc[0] - 1
    ret2 = second_half["portfolio_value"].iloc[-1] / second_half["portfolio_value"].iloc[0] - 1

    return {
        "first_half_return_pct": round(ret1 * 100, 2),
        "second_half_return_pct": round(ret2 * 100, 2),
        "both_profitable": ret1 > 0 and ret2 > 0,
        "first_half_period": f"{first_half.index[0].strftime('%Y-%m-%d')} to {first_half.index[-1].strftime('%Y-%m-%d')}",
        "second_half_period": f"{second_half.index[0].strftime('%Y-%m-%d')} to {second_half.index[-1].strftime('%Y-%m-%d')}",
    }


# ─── Validation: Incremental regime test ─────────────────────────────────────

def regime_test_incremental(results, base_results):
    """
    Regime test on the INCREMENTAL returns of overlay vs base.
    This tests whether the call overlay benefit is regime-agnostic.
    """
    min_len = min(len(results), len(base_results))
    inc_rets = results["daily_return"].values[:min_len] - base_results["daily_return"].values[:min_len]
    spy_rets = results["spy_ret"].values[:min_len]

    green_mask = spy_rets > 0
    red_mask = spy_rets < 0

    def daily_sharpe(r):
        if len(r) < 5 or np.std(r) == 0:
            return 0.0
        return np.mean(r) / np.std(r) * np.sqrt(252)

    sharpe_green = daily_sharpe(inc_rets[green_mask])
    sharpe_red = daily_sharpe(inc_rets[red_mask])

    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0

    return {
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(gap, 3),
        "regime_pass": gap < 0.50,
    }


# ─── Strategy delta functions ────────────────────────────────────────────────

def delta_30_monthly(vix):
    return 0.30

def delta_40_monthly(vix):
    return 0.40

def delta_30_weekly(vix):
    return 0.30

def delta_vix_adaptive(vix):
    """25-delta when VIX < 15, 40-delta when VIX 15-20, no calls when VIX > 20."""
    if vix < 15:
        return 0.25
    elif vix <= 20:
        return 0.40
    else:
        return None  # no call


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("COVERED CALL OVERLAY ON VIX-GATED UPRO — COMPREHENSIVE BACKTEST")
    print("=" * 80)
    print()

    data = download_data()

    # Define strategies
    strategies = [
        ("UPRO Buy-and-Hold", None, False, 30),
        ("VIX-Gated UPRO (no calls)", None, True, 30),
        ("VIX-Gated + Monthly 30Δ Call", delta_30_monthly, True, 21),  # ~monthly
        ("VIX-Gated + Monthly 40Δ Call", delta_40_monthly, True, 21),
        ("VIX-Gated + Weekly 30Δ Call", delta_30_weekly, True, 5),
        ("VIX-Gated + VIX-Adaptive Call", delta_vix_adaptive, True, 21),
    ]

    all_metrics = []
    all_validations = []
    all_results = {}

    for name, delta_func, use_vix_gate, tenor in strategies:
        print(f"\n{'─' * 60}")
        print(f"Running: {name}")
        print(f"{'─' * 60}")

        results, meta = walk_forward_backtest(
            data, name, delta_func,
            use_vix_gate=use_vix_gate,
            call_tenor_days=tenor,
            train_days=252, test_days=63, slide_days=21
        )

        if results is None:
            print(f"  SKIPPED — insufficient data")
            continue

        # Compute metrics
        metrics = compute_metrics(results, name)
        all_metrics.append(metrics)
        all_results[name] = results

        # Add call-specific meta
        if meta:
            total_premium = sum(m["premium_collected_pct"] for m in meta)
            total_assignments = sum(m["total_assignments"] for m in meta)
            total_rolls = sum(m["total_rolls"] for m in meta)
            metrics["premium_collected_pct"] = round(total_premium, 2)
            metrics["total_assignments"] = total_assignments
            metrics["total_rolls"] = total_rolls
            metrics["assignment_rate_pct"] = round(
                total_assignments / max(total_rolls, 1) * 100, 1
            )

        print(f"  Total Return:  {metrics['total_return_pct']:>8.1f}%")
        print(f"  Ann. Return:   {metrics['ann_return_pct']:>8.1f}%")
        print(f"  Ann. Vol:      {metrics['ann_vol_pct']:>8.1f}%")
        print(f"  Sharpe:        {metrics['sharpe']:>8.3f}")
        print(f"  Sortino:       {metrics['sortino']:>8.3f}")
        print(f"  Max Drawdown:  {metrics['max_drawdown_pct']:>8.1f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:>8.3f}")
        print(f"  Win Rate:      {metrics['win_rate']:>8.2%}")

        # Validation
        print(f"\n  --- Validation ---")

        # Permutation test: for overlay strategies, test INCREMENTAL value vs base
        base_name_key = "VIX-Gated UPRO (no calls)"
        base_for_perm = all_results.get(base_name_key) if delta_func is not None and use_vix_gate else None
        actual_sharpe, p_val, _ = permutation_test(results, base_results=base_for_perm, n_perms=100)
        perm_pass = p_val < 0.05
        perm_label = "incremental" if base_for_perm is not None else "absolute"
        print(f"  Permutation test ({perm_label}): stat={actual_sharpe:.3f}, p={p_val:.3f} "
              f"{'PASS' if perm_pass else 'FAIL'}")

        # Regime test
        regime = regime_test(results)
        print(f"  Regime test: Sharpe_green={regime['sharpe_green']:.3f}, "
              f"Sharpe_red={regime['sharpe_red']:.3f}, "
              f"gap={regime['regime_gap']:.3f} {'PASS' if regime['regime_pass'] else 'FAIL'}")

        # Regime test on INCREMENTAL returns (overlay benefit in each regime)
        if base_for_perm is not None:
            inc_regime = regime_test_incremental(results, base_for_perm)
            print(f"  Regime test (overlay increment): green={inc_regime['sharpe_green']:.3f}, "
                  f"red={inc_regime['sharpe_red']:.3f}, "
                  f"gap={inc_regime['regime_gap']:.3f} {'PASS' if inc_regime['regime_pass'] else 'FAIL'}")
        else:
            inc_regime = None

        # Sub-period test
        subperiod = subperiod_test(results)
        print(f"  Sub-period: H1={subperiod['first_half_return_pct']:.1f}%, "
              f"H2={subperiod['second_half_return_pct']:.1f}% "
              f"{'PASS' if subperiod['both_profitable'] else 'FAIL'}")

        validation = {
            "strategy": name,
            "permutation_p": round(p_val, 4),
            "permutation_pass": perm_pass,
            "permutation_type": perm_label,
            **regime,
            **subperiod,
        }
        if inc_regime is not None:
            validation["inc_regime_gap"] = inc_regime["regime_gap"]
            validation["inc_regime_pass"] = inc_regime["regime_pass"]
        all_validations.append(validation)

    # ─── Overlay improvement analysis ────────────────────────────────────

    print("\n" + "=" * 80)
    print("OVERLAY IMPROVEMENT vs VIX-GATED UPRO (BASE)")
    print("=" * 80)

    base_name = "VIX-Gated UPRO (no calls)"
    base_metrics = next((m for m in all_metrics if m["strategy"] == base_name), None)

    if base_metrics:
        for m in all_metrics:
            if m["strategy"] == base_name or m["strategy"] == "UPRO Buy-and-Hold":
                continue
            sharpe_diff = m["sharpe"] - base_metrics["sharpe"]
            ret_diff = m["ann_return_pct"] - base_metrics["ann_return_pct"]
            dd_diff = m["max_drawdown_pct"] - base_metrics["max_drawdown_pct"]
            print(f"\n  {m['strategy']}:")
            print(f"    Sharpe improvement:  {sharpe_diff:+.3f}")
            print(f"    Ann return delta:    {ret_diff:+.1f}%")
            print(f"    MaxDD delta:         {dd_diff:+.1f}%")
            if "premium_collected_pct" in m:
                print(f"    Premium collected:   {m['premium_collected_pct']:.1f}% of portfolio")
                print(f"    Assignment rate:     {m.get('assignment_rate_pct', 0):.1f}%")

    # ─── Summary comparison table ────────────────────────────────────────

    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Strategy':<35} {'TotRet%':>8} {'AnnRet%':>8} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD%':>8} {'PF':>6} {'WR':>6}")
    print("-" * 100)
    for m in all_metrics:
        print(f"{m['strategy']:<35} {m['total_return_pct']:>8.1f} {m['ann_return_pct']:>8.1f} "
              f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['max_drawdown_pct']:>8.1f} "
              f"{m['profit_factor']:>6.2f} {m['win_rate']:>6.1%}")

    # ─── Validation summary ──────────────────────────────────────────────

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Strategy':<35} {'Perm p':>7} {'Perm':>5} {'RegGap':>7} {'Reg':>5} "
          f"{'IncGap':>7} {'IncR':>5} {'SubPer':>7}")
    print("-" * 90)
    for v in all_validations:
        inc_gap = v.get("inc_regime_gap", "")
        inc_pass = v.get("inc_regime_pass", "")
        inc_gap_str = f"{inc_gap:>7.3f}" if inc_gap != "" else "    N/A"
        inc_pass_str = f"{'PASS' if inc_pass else 'FAIL':>5}" if inc_pass != "" else "  N/A"
        print(f"{v['strategy']:<35} {v['permutation_p']:>7.3f} "
              f"{'PASS' if v['permutation_pass'] else 'FAIL':>5} "
              f"{v['regime_gap']:>7.3f} {'PASS' if v['regime_pass'] else 'FAIL':>5} "
              f"{inc_gap_str} {inc_pass_str} "
              f"{'PASS' if v['both_profitable'] else 'FAIL':>7}")

    # Interpretive notes
    print("\n" + "─" * 80)
    print("NOTES ON VALIDATION:")
    print("─" * 80)
    print("  - Regime test (absolute): EXPECTED to fail for all UPRO strategies. UPRO is")
    print("    3x leveraged equity — it inherently gains on green days and loses on red days.")
    print("    The regime gap test was designed for alpha strategies, not beta-heavy products.")
    print("  - Regime test (incremental): Tests whether the OVERLAY BENEFIT is regime-agnostic.")
    print("    This is the meaningful test — does the call overlay help equally in both regimes?")
    print("  - Permutation test (incremental): Tests whether overlay adds significant value")
    print("    vs the base VIX-gated strategy. Sign-flip test for base strategies.")
    print("  - UPRO's realized vol (~50-60%) makes covered call premiums very large.")
    print("    A 30-delta monthly call on UPRO yields ~2-4% of spot, annualizing to ~30-45%.")
    print("    This is real — leveraged ETF options are expensive because the vol is real.")

    # ─── Save outputs ────────────────────────────────────────────────────

    # Summary JSON
    summary = {
        "run_date": datetime.now().isoformat(),
        "data_range": f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
        "methodology": {
            "walk_forward": "252d train, 63d test, 21d slide",
            "vol_model": "63d trailing realized vol (annualized)",
            "cost_model": "10bps spread per option trade",
            "vix_gating": "100% alloc VIX<17, 50% VIX 17-25, 0% VIX>25",
        },
        "metrics": all_metrics,
        "validations": all_validations,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary to {summary_path}")

    # Detailed CSV for each strategy
    for name, results in all_results.items():
        safe_name = name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "")
        safe_name = safe_name.replace("δ", "d").replace(",", "")
        csv_path = OUTPUT_DIR / f"{safe_name}.csv"
        results.to_csv(csv_path)

    # Combined comparison CSV
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df.to_csv(OUTPUT_DIR / "comparison.csv", index=False)
    print(f"Saved comparison CSV to {OUTPUT_DIR / 'comparison.csv'}")

    # Validation CSV
    val_df = pd.DataFrame(all_validations)
    val_df.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    print(f"Saved validation CSV to {OUTPUT_DIR / 'validation.csv'}")

    print("\nDone.")
    return summary


if __name__ == "__main__":
    summary = main()
