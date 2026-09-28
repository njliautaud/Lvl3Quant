#!/usr/bin/env python3
"""
Adversarial Validation: Vol-Adjusted Relative Strength Strategy
Assets: GLD, TLT, UUP
Selection: weekly (Friday), pick best 20d return / 20d vol
Vol target: 8% annualized, 0.02% slippage, $645 initial capital

6 checks:
1. Inverse selection (pick worst)
2. Always-gold baseline
3. Selection frequency
4. Sub-period stability
5. Parameter sensitivity grid
6. Cost sensitivity
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from itertools import product

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
TICKERS = ["GLD", "TLT", "UUP"]
START = "2022-01-01"
END = "2026-07-29"
INITIAL_CAPITAL = 645.0
TARGET_VOL = 0.08
SLIPPAGE_BPS = 0.0002  # 0.02%
LOOKBACK = 20
ANNUALIZE = np.sqrt(252)

RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/voladj_rs_adversarial_results.json"


# ── DATA ────────────────────────────────────────────────────────────────────
def download_data():
    print("Downloading data...")
    df = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    # Handle multi-level columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        prices = df["Close"]
    else:
        prices = df[TICKERS]
    prices = prices.dropna()
    print(f"  Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    return prices


# ── STRATEGY ENGINE ─────────────────────────────────────────────────────────
def run_strategy(prices, lookback=20, target_vol=0.08, slippage_bps=0.0002,
                 rebal_freq="weekly", selection="best", force_asset=None):
    """
    Run the vol-adjusted relative strength strategy.

    selection: "best" (normal), "worst" (inverse), or None if force_asset set
    force_asset: always hold this asset (e.g., "GLD")
    rebal_freq: "weekly", "biweekly", "monthly"
    """
    rets = prices.pct_change()

    # Compute risk-adjusted momentum for each asset
    momentum = pd.DataFrame(index=prices.index, columns=TICKERS, dtype=float)
    for t in TICKERS:
        roll_ret = rets[t].rolling(lookback).mean() * lookback  # cumulative-ish
        roll_vol = rets[t].rolling(lookback).std()
        momentum[t] = roll_ret / roll_vol
    momentum = momentum.dropna()

    # Align
    common_idx = momentum.index.intersection(rets.index)
    momentum = momentum.loc[common_idx]
    rets_aligned = rets.loc[common_idx]

    # Determine rebalance days
    dates = momentum.index.to_series()
    if rebal_freq == "weekly":
        rebal_mask = dates.dt.dayofweek == 4  # Friday
    elif rebal_freq == "biweekly":
        fridays = dates[dates.dt.dayofweek == 4]
        rebal_mask = pd.Series(False, index=dates.index)
        for i, d in enumerate(fridays):
            if i % 2 == 0:
                rebal_mask[d] = True
    elif rebal_freq == "monthly":
        # Last trading day of month
        rebal_mask = ~dates.dt.month.eq(dates.dt.month.shift(-1))
    else:
        rebal_mask = dates.dt.dayofweek == 4

    # Build holdings
    selected_asset = pd.Series(index=common_idx, dtype=str)
    current_asset = None

    for i, date in enumerate(common_idx):
        if rebal_mask.iloc[i] or current_asset is None:
            if force_asset:
                current_asset = force_asset
            elif selection == "best":
                current_asset = momentum.loc[date].idxmax()
            elif selection == "worst":
                current_asset = momentum.loc[date].idxmin()
        selected_asset.iloc[i] = current_asset

    # Compute strategy returns with vol targeting and slippage
    strat_rets = pd.Series(0.0, index=common_idx)
    prev_asset = None

    for i, date in enumerate(common_idx):
        asset = selected_asset.iloc[i]
        asset_ret = rets_aligned.loc[date, asset]

        # Vol targeting: scale position by target_vol / realized_vol
        realized_vol = rets[asset].loc[:date].tail(lookback).std() * ANNUALIZE
        if realized_vol > 0 and not np.isnan(realized_vol):
            vol_scale = min(target_vol / realized_vol, 2.0)  # cap leverage at 2x
        else:
            vol_scale = 1.0

        daily_ret = asset_ret * vol_scale

        # Apply slippage on rebalance
        if asset != prev_asset and prev_asset is not None:
            daily_ret -= slippage_bps * 2  # buy + sell

        strat_rets.iloc[i] = daily_ret
        prev_asset = asset

    return strat_rets, selected_asset


def compute_metrics(rets):
    """Compute Sharpe, Sortino, MaxDD, total return."""
    if len(rets) < 10 or rets.std() == 0:
        return {"sharpe": 0, "sortino": 0, "maxdd": 0, "total_return": 0, "ann_return": 0}

    cum = (1 + rets).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_years = len(rets) / 252
    ann_ret = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1

    sharpe = (rets.mean() / rets.std()) * ANNUALIZE if rets.std() > 0 else 0

    downside = rets[rets < 0].std()
    sortino = (rets.mean() / downside) * ANNUALIZE if downside > 0 else 0

    drawdown = cum / cum.cummax() - 1
    maxdd = drawdown.min()

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "maxdd": round(float(maxdd) * 100, 2),  # percentage
        "total_return": round(float(total_ret) * 100, 2),
        "ann_return": round(float(ann_ret) * 100, 2),
    }


# ── ADVERSARIAL CHECKS ─────────────────────────────────────────────────────
def check_1_inverse(prices):
    """Pick WORST risk-adjusted momentum. Should lose money if selection has edge."""
    print("\n=== CHECK 1: INVERSE SELECTION ===")
    inv_rets, _ = run_strategy(prices, selection="worst")
    norm_rets, _ = run_strategy(prices, selection="best")

    inv_m = compute_metrics(inv_rets)
    norm_m = compute_metrics(norm_rets)

    passed = inv_m["sharpe"] < norm_m["sharpe"] - 0.3  # inverse should be meaningfully worse

    print(f"  Normal Sharpe:  {norm_m['sharpe']}")
    print(f"  Inverse Sharpe: {inv_m['sharpe']}")
    print(f"  Delta: {norm_m['sharpe'] - inv_m['sharpe']:.3f}")
    print(f"  PASS: {passed} (inverse Sharpe < normal - 0.30)")

    return {
        "name": "Inverse Selection",
        "passed": bool(passed),
        "normal_sharpe": norm_m["sharpe"],
        "inverse_sharpe": inv_m["sharpe"],
        "normal_metrics": norm_m,
        "inverse_metrics": inv_m,
        "delta": round(norm_m["sharpe"] - inv_m["sharpe"], 3),
    }


def check_2_always_gold(prices):
    """Always hold GLD with same vol target. CRITICAL TEST."""
    print("\n=== CHECK 2: ALWAYS-GOLD BASELINE (CRITICAL) ===")
    gold_rets, _ = run_strategy(prices, force_asset="GLD")
    norm_rets, _ = run_strategy(prices, selection="best")

    gold_m = compute_metrics(gold_rets)
    norm_m = compute_metrics(norm_rets)

    delta = norm_m["sharpe"] - gold_m["sharpe"]
    passed = delta > 0.20  # strategy must beat gold-only by at least 0.20 Sharpe

    # Also check always-TLT and always-UUP
    tlt_rets, _ = run_strategy(prices, force_asset="TLT")
    uup_rets, _ = run_strategy(prices, force_asset="UUP")
    tlt_m = compute_metrics(tlt_rets)
    uup_m = compute_metrics(uup_rets)

    print(f"  Strategy Sharpe: {norm_m['sharpe']}")
    print(f"  Always-GLD Sharpe: {gold_m['sharpe']}")
    print(f"  Always-TLT Sharpe: {tlt_m['sharpe']}")
    print(f"  Always-UUP Sharpe: {uup_m['sharpe']}")
    print(f"  Strategy vs Gold delta: {delta:.3f}")
    print(f"  PASS: {passed} (strategy > gold + 0.20)")

    return {
        "name": "Always-Gold Baseline (CRITICAL)",
        "passed": bool(passed),
        "strategy_sharpe": norm_m["sharpe"],
        "always_gld_sharpe": gold_m["sharpe"],
        "always_tlt_sharpe": tlt_m["sharpe"],
        "always_uup_sharpe": uup_m["sharpe"],
        "delta_vs_gold": round(delta, 3),
        "strategy_metrics": norm_m,
        "gold_metrics": gold_m,
    }


def check_3_selection_frequency(prices):
    """Count how often each asset was selected."""
    print("\n=== CHECK 3: SELECTION FREQUENCY ===")
    _, selected = run_strategy(prices, selection="best")

    counts = selected.value_counts(normalize=True)
    freq = {t: round(float(counts.get(t, 0)) * 100, 1) for t in TICKERS}

    gld_pct = freq.get("GLD", 0)
    passed = gld_pct <= 70  # GLD should not dominate >70%

    print(f"  GLD: {freq.get('GLD', 0)}%")
    print(f"  TLT: {freq.get('TLT', 0)}%")
    print(f"  UUP: {freq.get('UUP', 0)}%")
    print(f"  PASS: {passed} (GLD <= 70%)")

    return {
        "name": "Selection Frequency",
        "passed": bool(passed),
        "frequencies": freq,
        "gld_dominated": gld_pct > 70,
    }


def check_4_subperiod_stability(prices):
    """Split into 4 equal sub-periods, all should have Sharpe > 0."""
    print("\n=== CHECK 4: SUB-PERIOD STABILITY ===")
    rets, _ = run_strategy(prices, selection="best")

    n = len(rets)
    chunk = n // 4
    sub_results = []
    all_positive = True

    for i in range(4):
        start_idx = i * chunk
        end_idx = (i + 1) * chunk if i < 3 else n
        sub = rets.iloc[start_idx:end_idx]
        m = compute_metrics(sub)
        period_start = sub.index[0].strftime("%Y-%m-%d")
        period_end = sub.index[-1].strftime("%Y-%m-%d")

        if m["sharpe"] <= 0:
            all_positive = False

        sub_results.append({
            "period": f"{period_start} to {period_end}",
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "total_return": m["total_return"],
        })
        print(f"  Period {i+1} ({period_start} to {period_end}): Sharpe={m['sharpe']}, Return={m['total_return']}%")

    print(f"  PASS: {all_positive} (all 4 periods Sharpe > 0)")

    return {
        "name": "Sub-Period Stability",
        "passed": bool(all_positive),
        "sub_periods": sub_results,
    }


def check_5_parameter_sensitivity(prices):
    """Grid search across lookback, rebal_freq, target_vol."""
    print("\n=== CHECK 5: PARAMETER SENSITIVITY ===")
    lookbacks = [5, 10, 15, 20, 30, 40, 60]
    rebal_freqs = ["weekly", "biweekly", "monthly"]
    target_vols = [0.06, 0.08, 0.10, 0.12]

    total = len(lookbacks) * len(rebal_freqs) * len(target_vols)
    passing = 0
    best_sharpe = -999
    worst_sharpe = 999
    all_sharpes = []

    for lb, rf, tv in product(lookbacks, rebal_freqs, target_vols):
        try:
            r, _ = run_strategy(prices, lookback=lb, target_vol=tv, rebal_freq=rf, selection="best")
            m = compute_metrics(r)
            s = m["sharpe"]
            all_sharpes.append({"lookback": lb, "rebal": rf, "target_vol": tv, "sharpe": s})
            if s > 0.3:
                passing += 1
            best_sharpe = max(best_sharpe, s)
            worst_sharpe = min(worst_sharpe, s)
        except Exception:
            all_sharpes.append({"lookback": lb, "rebal": rf, "target_vol": tv, "sharpe": None})

    pct_passing = passing / total * 100
    passed = pct_passing >= 50  # at least 50% of combos should work

    # Median sharpe
    valid_sharpes = [x["sharpe"] for x in all_sharpes if x["sharpe"] is not None]
    median_sharpe = float(np.median(valid_sharpes)) if valid_sharpes else 0

    print(f"  {passing}/{total} combos have Sharpe > 0.3 ({pct_passing:.1f}%)")
    print(f"  Median Sharpe: {median_sharpe:.3f}")
    print(f"  Best Sharpe: {best_sharpe:.3f}, Worst: {worst_sharpe:.3f}")
    print(f"  PASS: {passed} (>= 50% of combos Sharpe > 0.3)")

    return {
        "name": "Parameter Sensitivity",
        "passed": bool(passed),
        "total_combos": total,
        "passing_combos": passing,
        "pct_passing": round(pct_passing, 1),
        "median_sharpe": round(median_sharpe, 3),
        "best_sharpe": round(best_sharpe, 3),
        "worst_sharpe": round(worst_sharpe, 3),
    }


def check_6_cost_sensitivity(prices):
    """Run at higher slippage levels."""
    print("\n=== CHECK 6: COST SENSITIVITY ===")
    slippage_levels = [0.0005, 0.0010, 0.0015, 0.0020]  # 5, 10, 15, 20 bps
    results = []

    survives_10bps = False
    for slip in slippage_levels:
        r, _ = run_strategy(prices, slippage_bps=slip, selection="best")
        m = compute_metrics(r)
        label = f"{slip*10000:.0f}bps"
        results.append({"slippage": label, "sharpe": m["sharpe"], "total_return": m["total_return"]})
        print(f"  Slippage {label}: Sharpe={m['sharpe']}, Return={m['total_return']}%")
        if slip == 0.0010 and m["sharpe"] > 0.3:
            survives_10bps = True

    print(f"  PASS: {survives_10bps} (Sharpe > 0.3 at 10bps slippage)")

    return {
        "name": "Cost Sensitivity",
        "passed": bool(survives_10bps),
        "levels": results,
    }


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: Vol-Adjusted Relative Strength")
    print(f"Assets: {TICKERS} | Lookback: {LOOKBACK}d | Vol Target: {TARGET_VOL*100}%")
    print(f"Capital: ${INITIAL_CAPITAL} | Slippage: {SLIPPAGE_BPS*10000}bps")
    print("=" * 70)

    prices = download_data()

    # Run all 6 checks
    c1 = check_1_inverse(prices)
    c2 = check_2_always_gold(prices)
    c3 = check_3_selection_frequency(prices)
    c4 = check_4_subperiod_stability(prices)
    c5 = check_5_parameter_sensitivity(prices)
    c6 = check_6_cost_sensitivity(prices)

    checks = [c1, c2, c3, c4, c5, c6]
    n_passed = sum(1 for c in checks if c["passed"])
    overall_pass = n_passed >= 5

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)
    for i, c in enumerate(checks, 1):
        status = "PASS" if c["passed"] else "FAIL"
        critical = " ** CRITICAL **" if i == 2 else ""
        print(f"  Check {i}: {c['name']} — {status}{critical}")

    print(f"\n  OVERALL: {n_passed}/6 checks passed — {'PASS' if overall_pass else 'FAIL'}")

    if not c2["passed"]:
        print("\n  *** CRITICAL FAILURE: Strategy does NOT meaningfully beat always-gold. ***")
        print(f"  *** The 'selection' is likely just gold exposure. Delta vs gold: {c2['delta_vs_gold']:.3f} Sharpe ***")

    # Diagnosis
    diagnosis = []
    if c3["frequencies"].get("GLD", 0) > 60:
        diagnosis.append(f"GLD selected {c3['frequencies']['GLD']}% of the time - heavy gold bias")
    if not c2["passed"]:
        diagnosis.append("Strategy does not beat always-gold by 0.20+ Sharpe - edge is gold exposure, not rotation")
    if not c1["passed"]:
        diagnosis.append("Inverse selection is not significantly worse - selection signal is weak")
    if c5["pct_passing"] < 50:
        diagnosis.append(f"Only {c5['pct_passing']}% of parameter combos work - fragile/overfit")

    if diagnosis:
        print("\n  DIAGNOSIS:")
        for d in diagnosis:
            print(f"    - {d}")

    # Save results
    output = {
        "strategy": "Vol-Adjusted Relative Strength",
        "assets": TICKERS,
        "period": f"{START} to {END}",
        "baseline_params": {
            "lookback": LOOKBACK,
            "target_vol": TARGET_VOL,
            "slippage_bps": SLIPPAGE_BPS * 10000,
            "initial_capital": INITIAL_CAPITAL,
            "rebal_freq": "weekly",
        },
        "timestamp": datetime.now().isoformat(),
        "overall_pass": overall_pass,
        "checks_passed": n_passed,
        "checks_total": 6,
        "checks": checks,
        "diagnosis": diagnosis,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    main()
