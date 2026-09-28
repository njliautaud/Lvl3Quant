#!/usr/bin/env python3
"""
Sector Rotation Based on Economic Cycle v1
============================================
Use leading indicators to identify economic cycle phase, then rotate
between cyclical sectors (expansion) and defensive sectors (contraction).

Leading indicators (proxied via ETFs/indices):
- ISM PMI proxy: XLI (industrials) relative strength vs XLP (staples)
- Initial claims proxy: inverse momentum of employment-sensitive sectors
- Building permits proxy: ITB (homebuilders ETF) momentum
- Copper/Gold ratio: CPER/GLD (expansion/contraction signal)

Phases:
1. EXPANSION: Overweight XLK, XLY, XLI, XLF (cyclicals)
2. LATE CYCLE: Overweight XLE, XLB (commodities benefit)
3. CONTRACTION: Overweight XLU, XLP, XLV (defensives)
4. RECOVERY: Overweight XLF, XLI, XLY (early cyclicals)

Monthly rebalance. $100K initial, no DCA.

HC #0  : Sliding walk-forward (trailing lookback)
HC #428: Regime-agnostic validation (R1)
HC #694: Commission-free (Robinhood)
HC #697: No crypto
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/econ_cycle_rotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2010-01-01"
END = "2026-07-17"
INITIAL_CAPITAL = 100_000

# Sector ETFs
CYCLICAL_SECTORS = ["XLK", "XLY", "XLI", "XLF"]
LATE_CYCLE_SECTORS = ["XLE", "XLB"]
DEFENSIVE_SECTORS = ["XLU", "XLP", "XLV"]
RECOVERY_SECTORS = ["XLF", "XLI", "XLY"]

ALL_SECTORS = list(set(CYCLICAL_SECTORS + LATE_CYCLE_SECTORS + DEFENSIVE_SECTORS + RECOVERY_SECTORS))

# Leading indicator proxies
INDICATOR_TICKERS = ["XLI", "XLP", "ITB", "GLD", "SPY", "^VIX"]
# CPER (copper ETF) has limited history, use DBC as partial proxy
COPPER_PROXY = "DBC"  # Commodities ETF as copper proxy

REBAL_FREQ = "ME"
LOOKBACK = 63  # 3-month lookback for indicators


def download_data():
    """Download all sector and indicator data."""
    tickers = ALL_SECTORS + INDICATOR_TICKERS + [COPPER_PROXY, "SHY", "HYG", "IEF"]
    tickers = list(set(tickers))

    print(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    prices = prices.ffill(limit=5)

    vix = prices["^VIX"].copy() if "^VIX" in prices.columns else None
    prices_clean = prices.drop(columns=["^VIX"], errors="ignore")

    print(f"  Data: {prices_clean.shape[0]} days, {prices_clean.shape[1]} tickers")
    print(f"  Range: {prices_clean.index[0].date()} to {prices_clean.index[-1].date()}")

    return prices_clean, vix


def classify_economic_phase(prices, vix, date, lookback=LOOKBACK):
    """
    Classify current economic phase based on leading indicators.

    Returns: one of "expansion", "late_cycle", "contraction", "recovery"

    Scoring system:
    - XLI/XLP ratio trending up = expansion signal
    - ITB momentum positive = expansion signal
    - Copper/Gold (DBC/GLD) ratio trending up = expansion signal
    - VIX trending down = expansion signal
    - HYG/IEF ratio trending up = credit easing = expansion signal
    """
    loc = prices.index.get_loc(date)
    if loc < lookback + 10:
        return "neutral"

    window = prices.iloc[max(0, loc - lookback):loc + 1]

    scores = {"expansion": 0, "contraction": 0, "late_cycle": 0, "recovery": 0}

    # 1. XLI/XLP ratio (industrials vs staples) — expansion indicator
    if "XLI" in window.columns and "XLP" in window.columns:
        ratio = window["XLI"] / window["XLP"]
        ratio_mom = ratio.iloc[-1] / ratio.iloc[0] - 1
        ratio_trend = ratio.iloc[-1] > ratio.rolling(21).mean().iloc[-1]
        if ratio_mom > 0.02 and ratio_trend:
            scores["expansion"] += 2
        elif ratio_mom < -0.02:
            scores["contraction"] += 2

    # 2. ITB (homebuilders) momentum — leading indicator
    if "ITB" in window.columns:
        itb = window["ITB"]
        itb_mom = itb.iloc[-1] / itb.iloc[0] - 1
        itb_ma = itb.iloc[-1] > itb.rolling(50, min_periods=30).mean().iloc[-1]
        if itb_mom > 0.05 and itb_ma:
            scores["expansion"] += 1
            scores["recovery"] += 1
        elif itb_mom < -0.05:
            scores["contraction"] += 1
            scores["late_cycle"] += 1

    # 3. Copper/Gold proxy (DBC/GLD) — expansion/contraction
    if COPPER_PROXY in window.columns and "GLD" in window.columns:
        cg_ratio = window[COPPER_PROXY] / window["GLD"]
        cg_mom = cg_ratio.iloc[-1] / cg_ratio.iloc[0] - 1
        if cg_mom > 0.03:
            scores["expansion"] += 2
            scores["late_cycle"] += 1
        elif cg_mom < -0.03:
            scores["contraction"] += 2

    # 4. VIX trend
    if vix is not None:
        v = vix.iloc[max(0, loc - lookback):loc + 1]
        if len(v) > 10:
            vix_mom = v.iloc[-1] / v.iloc[0] - 1
            vix_level = v.iloc[-1]
            if vix_mom < -0.1 and vix_level < 20:
                scores["expansion"] += 1
            elif vix_level > 25:
                scores["contraction"] += 2
            elif vix_mom > 0.2:
                scores["late_cycle"] += 1

    # 5. Credit conditions: HYG/IEF ratio
    if "HYG" in window.columns and "IEF" in window.columns:
        credit = window["HYG"] / window["IEF"]
        credit_mom = credit.iloc[-1] / credit.iloc[0] - 1
        if credit_mom > 0.02:
            scores["expansion"] += 1
            scores["recovery"] += 1
        elif credit_mom < -0.02:
            scores["contraction"] += 1

    # 6. SPY trend — broad market health
    if "SPY" in window.columns:
        spy = window["SPY"]
        spy_above_ma = spy.iloc[-1] > spy.rolling(50, min_periods=30).mean().iloc[-1]
        spy_mom = spy.iloc[-1] / spy.iloc[0] - 1
        if spy_above_ma and spy_mom > 0.03:
            scores["expansion"] += 1
        elif not spy_above_ma and spy_mom < -0.03:
            scores["contraction"] += 1

    # Recovery detection: after contraction, early signs of improvement
    # Check if we were recently in contraction but indicators turning
    if scores["contraction"] >= 3 and scores["expansion"] >= 2:
        scores["recovery"] += 2

    # Pick the phase with highest score
    best_phase = max(scores, key=scores.get)
    best_score = scores[best_phase]

    # If no clear signal, default to neutral (equal weight all)
    if best_score <= 1:
        return "neutral"

    return best_phase


def get_phase_weights(phase):
    """Get sector allocation weights for each economic phase."""
    allocations = {
        "expansion": {
            "XLK": 0.30, "XLY": 0.25, "XLI": 0.25, "XLF": 0.20,
        },
        "late_cycle": {
            "XLE": 0.35, "XLB": 0.25, "XLK": 0.20, "XLP": 0.20,
        },
        "contraction": {
            "XLU": 0.30, "XLP": 0.35, "XLV": 0.35,
        },
        "recovery": {
            "XLF": 0.30, "XLI": 0.30, "XLY": 0.25, "XLK": 0.15,
        },
        "neutral": {
            # Equal weight across all unique sectors
            t: 1.0 / len(ALL_SECTORS) for t in ALL_SECTORS
        },
    }
    return allocations.get(phase, allocations["neutral"])


def backtest_econ_rotation(prices, vix, variant="base"):
    """
    Run economic cycle rotation backtest.

    Variants:
    - "base": Full phase detection with sector rotation
    - "simple_binary": Just cyclical vs defensive (simpler)
    - "with_leverage": Add SHY cash buffer in contraction, UPRO tilt in expansion
    - "equal_weight_sectors": Equal-weight all sectors always (benchmark)
    """
    returns = prices.pct_change().fillna(0)

    rebal_dates = pd.Series(range(len(prices.index)), index=prices.index).resample(REBAL_FREQ).last().index

    capital = INITIAL_CAPITAL
    equity_curve = []
    phase_log = []
    current_weights = {t: 1.0 / len(ALL_SECTORS) for t in ALL_SECTORS}

    warmup_date = prices.index[LOOKBACK + 60]

    for date in prices.index:
        if date < warmup_date:
            equity_curve.append({"date": date, "equity": capital, "phase": "warmup"})
            continue

        is_rebal = date in rebal_dates

        if is_rebal:
            if variant == "equal_weight_sectors":
                phase = "equal"
                current_weights = {t: 1.0 / len(ALL_SECTORS) for t in ALL_SECTORS}
            elif variant == "simple_binary":
                phase = classify_economic_phase(prices, vix, date)
                if phase in ["contraction", "late_cycle"]:
                    # Defensive
                    current_weights = {t: 0 for t in ALL_SECTORS}
                    for t in DEFENSIVE_SECTORS:
                        current_weights[t] = 1.0 / len(DEFENSIVE_SECTORS)
                else:
                    # Cyclical
                    current_weights = {t: 0 for t in ALL_SECTORS}
                    for t in CYCLICAL_SECTORS:
                        current_weights[t] = 1.0 / len(CYCLICAL_SECTORS)
            else:
                phase = classify_economic_phase(prices, vix, date)
                phase_weights = get_phase_weights(phase)
                current_weights = {t: 0 for t in ALL_SECTORS}
                current_weights.update(phase_weights)

            phase_log.append({"date": str(date.date()), "phase": phase})

        # Compute daily return
        daily_ret = 0
        for ticker, weight in current_weights.items():
            if weight > 0 and ticker in returns.columns and date in returns.index:
                daily_ret += weight * returns.loc[date, ticker]

        capital *= (1 + daily_ret)
        phase_name = phase_log[-1]["phase"] if phase_log else "warmup"
        equity_curve.append({"date": date, "equity": capital, "phase": phase_name})

    equity_df = pd.DataFrame(equity_curve).set_index("date")
    return equity_df, phase_log


def compute_metrics(equity_df, label="Strategy"):
    """Standard performance metrics."""
    eq = equity_df["equity"]
    daily_ret = eq.pct_change().dropna()

    total_return = eq.iloc[-1] / eq.iloc[0] - 1
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd = dd.min()

    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = (daily_ret.mean() * 252) / ann_vol if ann_vol > 0 else 0
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = (daily_ret.mean() * 252) / downside if downside > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "label": label,
        "total_return": float(total_return),
        "cagr": float(cagr),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_dd": float(max_dd),
        "calmar": float(calmar),
        "years": float(years),
        "final_equity": float(eq.iloc[-1]),
    }


def adversarial_validation(equity_df, spy_prices, n_perms=1000):
    """Adversarial validation suite."""
    results = {}
    eq = equity_df["equity"]
    daily_ret = eq.pct_change().dropna()
    actual_sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0

    # 1. Permutation test
    print("  Permutation test (1000 shuffles)...")
    perm_sharpes = []
    ret_vals = daily_ret.values.copy()
    for _ in range(n_perms):
        np.random.shuffle(ret_vals)
        s = (ret_vals.mean() * 252) / (ret_vals.std() * np.sqrt(252))
        perm_sharpes.append(s)
    p_value = (np.array(perm_sharpes) >= actual_sharpe).mean()
    results["permutation"] = {
        "actual_sharpe": float(actual_sharpe),
        "p_value": float(p_value),
        "pass": p_value < 0.05,
    }

    # 2. Sub-period stability (thirds)
    n = len(daily_ret)
    thirds = [daily_ret.iloc[:n//3], daily_ret.iloc[n//3:2*n//3], daily_ret.iloc[2*n//3:]]
    third_sharpes = []
    for t in thirds:
        s = (t.mean() * 252) / (t.std() * np.sqrt(252)) if t.std() > 0 else 0
        third_sharpes.append(float(s))
    results["sub_period"] = {
        "third_sharpes": third_sharpes,
        "all_positive": all(s > 0 for s in third_sharpes),
        "pass": all(s > 0 for s in third_sharpes),
    }

    # 3. Outlier removal
    sorted_ret = daily_ret.sort_values(ascending=False)
    trimmed = sorted_ret.iloc[10:]
    trimmed_sharpe = (trimmed.mean() * 252) / (trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
    trimmed_total = (1 + trimmed).prod() - 1
    results["outlier_removal"] = {
        "trimmed_sharpe": float(trimmed_sharpe),
        "trimmed_total_return": float(trimmed_total),
        "pass": trimmed_total > 0 and trimmed_sharpe > 0,
    }

    # 4. R1 Regime-agnostic (HC #428)
    spy_ret = spy_prices.pct_change().dropna()
    common = daily_ret.index.intersection(spy_ret.index)
    strat_m = daily_ret.reindex(common).dropna().resample("ME").sum()
    spy_m = spy_ret.reindex(common).dropna().resample("ME").sum()
    cm = strat_m.index.intersection(spy_m.index)

    green = strat_m.loc[cm][spy_m.loc[cm] > 0]
    red = strat_m.loc[cm][spy_m.loc[cm] <= 0]

    s_green = (green.mean() * 12) / (green.std() * np.sqrt(12)) if len(green) > 2 and green.std() > 0 else 0
    s_red = (red.mean() * 12) / (red.std() * np.sqrt(12)) if len(red) > 2 and red.std() > 0 else 0
    regime_ratio = abs(s_green - s_red) / max(abs(s_green), abs(s_red), 0.01)
    results["regime_agnostic"] = {
        "green_sharpe": float(s_green),
        "red_sharpe": float(s_red),
        "regime_ratio": float(regime_ratio),
        "pass": regime_ratio < 0.50,
    }

    # 5. Phase accuracy analysis
    # Check if our phase classifications are better than random
    if "phase" in equity_df.columns:
        phase_returns = {}
        for phase in ["expansion", "contraction", "late_cycle", "recovery", "neutral"]:
            mask = equity_df["phase"] == phase
            if mask.sum() > 20:
                phase_ret = daily_ret.reindex(equity_df.index[mask]).dropna()
                if len(phase_ret) > 5:
                    phase_returns[phase] = {
                        "mean_daily_ret": float(phase_ret.mean()),
                        "ann_sharpe": float((phase_ret.mean() * 252) / (phase_ret.std() * np.sqrt(252))) if phase_ret.std() > 0 else 0,
                        "days": int(mask.sum()),
                    }
        results["phase_analysis"] = phase_returns

    overall = all(r.get("pass", True) for k, r in results.items() if isinstance(r, dict) and "pass" in r)
    results["overall_pass"] = overall

    return results


def main():
    print("=" * 70)
    print("ECONOMIC CYCLE SECTOR ROTATION v1")
    print("=" * 70)

    prices, vix = download_data()

    all_results = {}
    equity_curves = {}

    for variant in ["base", "simple_binary", "equal_weight_sectors"]:
        print(f"\n--- Variant: {variant} ---")
        equity_df, phase_log = backtest_econ_rotation(prices, vix, variant=variant)
        metrics = compute_metrics(equity_df, label=f"EconRot_{variant}")

        print(f"  CAGR: {metrics['cagr']:.2%}")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  MaxDD: {metrics['max_dd']:.2%}")
        print(f"  Calmar: {metrics['calmar']:.3f}")
        print(f"  Final: ${metrics['final_equity']:,.0f}")

        # Phase distribution
        if phase_log:
            phases = pd.Series([p["phase"] for p in phase_log])
            print(f"  Phase distribution:")
            for phase, count in phases.value_counts().items():
                print(f"    {phase}: {count} months ({count/len(phases):.0%})")

        # Adversarial validation
        print(f"\n  Adversarial validation:")
        adv = adversarial_validation(equity_df, prices["SPY"])
        for test_name, test_result in adv.items():
            if isinstance(test_result, dict) and "pass" in test_result:
                status = "PASS" if test_result["pass"] else "FAIL"
                print(f"    {test_name}: {status}")
            elif test_name == "phase_analysis":
                print(f"    Phase breakdown:")
                for phase, pdata in test_result.items():
                    print(f"      {phase}: Sharpe {pdata['ann_sharpe']:.2f} ({pdata['days']} days)")
        print(f"    OVERALL: {'PASS' if adv.get('overall_pass') else 'FAIL'}")

        all_results[variant] = {
            "metrics": metrics,
            "adversarial": adv,
            "phase_log": phase_log[:24],  # First 24 months for reference
        }
        equity_curves[variant] = equity_df

    # SPY benchmark
    spy_eq = pd.DataFrame({"equity": INITIAL_CAPITAL * prices["SPY"].dropna() / prices["SPY"].dropna().iloc[0]})
    spy_metrics = compute_metrics(spy_eq, "SPY_BH")
    all_results["spy_benchmark"] = spy_metrics
    print(f"\n--- SPY Buy & Hold ---")
    print(f"  CAGR: {spy_metrics['cagr']:.2%}, Sharpe: {spy_metrics['sharpe']:.3f}, MaxDD: {spy_metrics['max_dd']:.2%}")

    # Save
    with open(OUTPUT_DIR / "econ_cycle_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    for variant, eq_df in equity_curves.items():
        axes[0].plot(eq_df.index, eq_df["equity"].values, label=variant, linewidth=1.2)
    axes[0].plot(spy_eq.index, spy_eq["equity"].values, label="SPY B&H", linestyle="--", alpha=0.7)
    axes[0].set_title("Economic Cycle Sector Rotation")
    axes[0].set_ylabel("Portfolio Value ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    # Phase timeline
    if equity_curves.get("base") is not None:
        eq = equity_curves["base"]
        phase_colors = {
            "expansion": "green", "late_cycle": "orange",
            "contraction": "red", "recovery": "blue", "neutral": "gray", "warmup": "lightgray",
        }
        if "phase" in eq.columns:
            for phase in eq["phase"].unique():
                mask = eq["phase"] == phase
                axes[1].fill_between(
                    eq.index, 0, 1,
                    where=mask, alpha=0.3,
                    color=phase_colors.get(phase, "gray"),
                    label=phase,
                )
            axes[1].set_title("Economic Phase Classification")
            axes[1].legend(loc="upper left")
            axes[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "econ_cycle_equity.png", dpi=150)
    plt.close()

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
