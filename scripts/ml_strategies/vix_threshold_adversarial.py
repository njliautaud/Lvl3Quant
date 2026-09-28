"""
VIX Threshold Strategy — Adversarial Validation
================================================
Tests: "Buy SPY when VIX > threshold, hold N days"
vs contrarian: "Buy SPY when VIX < threshold"

Adversarial gates:
1. Permutation test (100 shuffles, p < 0.05)
2. Regime gap (green/red months, gap < 0.50)
3. Sub-period consistency (3 blocks, CV of Sharpe < 0.50)
4. Outlier dependency (remove top 5% returns, degradation < 0.50)
"""

import json
import os
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/vix_threshold_adversarial")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COST_BPS = 2  # 2bps per trade
VIX_THRESHOLDS = [18, 20, 22, 25, 30]
HOLD_DAYS = [3, 5, 10, 20]
N_PERMUTATIONS = 100


def download_data():
    """Download VIX and SPY data 2008-2024."""
    print("Downloading SPY and VIX data...")
    spy = yf.download("SPY", start="2008-01-01", end="2024-12-31", progress=False)
    vix = yf.download("^VIX", start="2008-01-01", end="2024-12-31", progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    df = pd.DataFrame(index=spy.index)
    df['spy_close'] = spy['Close']
    df['spy_ret'] = spy['Close'].pct_change()
    df['vix_close'] = vix['Close'].reindex(spy.index, method='ffill')
    df = df.dropna()
    print(f"  Data: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')} ({len(df)} days)")
    return df


def calc_sharpe(returns, annual=True):
    """Annualized Sharpe ratio."""
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    if annual:
        return float(returns.mean() / returns.std() * np.sqrt(252))
    return float(returns.mean() / returns.std())


def calc_cagr(returns):
    """CAGR from daily returns series."""
    total = (1 + returns).prod()
    years = len(returns) / 252
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def calc_max_dd(returns):
    """Max drawdown from daily returns."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())


def run_strategy(df, vix_threshold, hold_days, direction="above"):
    """
    Run VIX threshold strategy.
    direction='above': buy when VIX > threshold (vol spike mean-reversion)
    direction='below': buy when VIX < threshold (calm markets stay invested)
    """
    signals = np.zeros(len(df))
    positions = np.zeros(len(df))

    vix = df['vix_close'].values

    # Generate entry signals
    if direction == "above":
        signals = (vix > vix_threshold).astype(float)
    else:
        signals = (vix < vix_threshold).astype(float)

    # Hold for N days after signal
    in_position = False
    hold_counter = 0
    trades = 0

    for i in range(len(df)):
        if not in_position and signals[i] == 1:
            in_position = True
            hold_counter = hold_days
            trades += 1
            positions[i] = 1
        elif in_position:
            hold_counter -= 1
            if hold_counter <= 0:
                in_position = False
                positions[i] = 0
            else:
                positions[i] = 1
        else:
            positions[i] = 0

    # Calculate returns with costs
    spy_ret = df['spy_ret'].values
    strat_ret = positions[:-1] * spy_ret[1:]  # position today -> return tomorrow

    # Apply costs on position changes
    pos_changes = np.diff(positions)
    cost_array = np.abs(pos_changes) * (COST_BPS / 10000)
    strat_ret = strat_ret - cost_array[:len(strat_ret)]

    strat_ret = pd.Series(strat_ret, index=df.index[1:len(strat_ret)+1])

    return strat_ret, trades, positions


def permutation_test(df, vix_threshold, hold_days, direction, actual_sharpe):
    """Shuffle VIX signal dates, compute Sharpe distribution."""
    shuffled_sharpes = []
    for _ in range(N_PERMUTATIONS):
        df_shuffled = df.copy()
        df_shuffled['vix_close'] = np.random.permutation(df_shuffled['vix_close'].values)
        ret, _, _ = run_strategy(df_shuffled, vix_threshold, hold_days, direction)
        shuffled_sharpes.append(calc_sharpe(ret))

    p_value = np.mean([s >= actual_sharpe for s in shuffled_sharpes])
    return float(p_value), shuffled_sharpes


def regime_gap_test(df, strat_ret):
    """Stratify by SPY regime (green/red month). Check gap < 0.50."""
    # Monthly SPY returns to classify regime
    monthly_spy = df['spy_ret'].resample('M').sum()

    # Map each day to its month's regime
    strat_ret_df = strat_ret.to_frame('ret')
    strat_ret_df['month'] = strat_ret_df.index.to_period('M')

    green_months = set(monthly_spy[monthly_spy > 0].index.to_period('M'))
    red_months = set(monthly_spy[monthly_spy <= 0].index.to_period('M'))

    green_ret = strat_ret_df[strat_ret_df['month'].isin(green_months)]['ret']
    red_ret = strat_ret_df[strat_ret_df['month'].isin(red_months)]['ret']

    sharpe_green = calc_sharpe(green_ret) if len(green_ret) > 20 else 0
    sharpe_red = calc_sharpe(red_ret) if len(red_ret) > 20 else 0

    max_abs = max(abs(sharpe_green), abs(sharpe_red))
    gap = abs(sharpe_green - sharpe_red) / max_abs if max_abs > 0 else 0

    return float(gap), float(sharpe_green), float(sharpe_red)


def subperiod_consistency(strat_ret):
    """Split into 3 equal blocks, CV of Sharpe < 0.50."""
    n = len(strat_ret)
    third = n // 3
    blocks = [strat_ret.iloc[:third], strat_ret.iloc[third:2*third], strat_ret.iloc[2*third:]]

    sharpes = [calc_sharpe(b) for b in blocks]
    mean_s = np.mean(sharpes)
    std_s = np.std(sharpes)
    cv = std_s / abs(mean_s) if abs(mean_s) > 0.01 else 999

    return float(cv), sharpes


def outlier_dependency(strat_ret, actual_sharpe):
    """Remove top 5% returns, check degradation < 0.50."""
    threshold = strat_ret.quantile(0.95)
    filtered = strat_ret[strat_ret <= threshold]
    filtered_sharpe = calc_sharpe(filtered)

    degradation = (actual_sharpe - filtered_sharpe) / abs(actual_sharpe) if abs(actual_sharpe) > 0.01 else 0
    return float(degradation), float(filtered_sharpe)


def evaluate_config(df, vix_threshold, hold_days, direction):
    """Full evaluation of one config."""
    strat_ret, trades, positions = run_strategy(df, vix_threshold, hold_days, direction)

    sharpe = calc_sharpe(strat_ret)
    cagr = calc_cagr(strat_ret)
    max_dd = calc_max_dd(strat_ret)

    # Exposure
    exposure = np.mean(positions)

    # Gate 1: Permutation test
    p_value, _ = permutation_test(df, vix_threshold, hold_days, direction, sharpe)
    perm_pass = p_value < 0.05

    # Gate 2: Regime gap
    regime_gap, sharpe_green, sharpe_red = regime_gap_test(df, strat_ret)
    regime_pass = regime_gap < 0.50

    # Gate 3: Sub-period consistency
    cv, block_sharpes = subperiod_consistency(strat_ret)
    consistency_pass = cv < 0.50

    # Gate 4: Outlier dependency
    degradation, filtered_sharpe = outlier_dependency(strat_ret, sharpe)
    outlier_pass = degradation < 0.50

    all_pass = perm_pass and regime_pass and consistency_pass and outlier_pass

    return {
        "direction": direction,
        "vix_threshold": vix_threshold,
        "hold_days": hold_days,
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "trades": trades,
        "exposure_pct": round(exposure * 100, 1),
        "gates": {
            "permutation": {"p_value": round(p_value, 3), "pass": perm_pass},
            "regime_gap": {
                "gap": round(regime_gap, 3),
                "sharpe_green": round(sharpe_green, 3),
                "sharpe_red": round(sharpe_red, 3),
                "pass": regime_pass
            },
            "subperiod_consistency": {
                "cv": round(cv, 3),
                "block_sharpes": [round(s, 3) for s in block_sharpes],
                "pass": consistency_pass
            },
            "outlier_dependency": {
                "degradation": round(degradation, 3),
                "filtered_sharpe": round(filtered_sharpe, 3),
                "pass": outlier_pass
            }
        },
        "all_gates_pass": all_pass
    }


def main():
    np.random.seed(42)
    df = download_data()

    results = {
        "metadata": {
            "run_date": datetime.now().isoformat(),
            "data_range": f"{df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}",
            "n_days": len(df),
            "cost_bps": COST_BPS,
            "n_permutations": N_PERMUTATIONS,
            "thresholds_tested": VIX_THRESHOLDS,
            "hold_days_tested": HOLD_DAYS
        },
        "buy_and_hold_spy": {},
        "vix_above_results": [],
        "vix_below_results": [],
        "summary": {}
    }

    # Buy and hold benchmark
    bh_ret = df['spy_ret'].iloc[1:]
    results["buy_and_hold_spy"] = {
        "sharpe": round(calc_sharpe(bh_ret), 3),
        "cagr": round(calc_cagr(bh_ret) * 100, 2),
        "max_dd": round(calc_max_dd(bh_ret) * 100, 2)
    }
    print(f"\nBuy & Hold SPY: Sharpe={results['buy_and_hold_spy']['sharpe']}, "
          f"CAGR={results['buy_and_hold_spy']['cagr']}%, MaxDD={results['buy_and_hold_spy']['max_dd']}%")

    # Test VIX ABOVE (buy when VIX is high — vol spike mean-reversion)
    print("\n" + "="*70)
    print("TESTING: Buy SPY when VIX ABOVE threshold (mean-reversion hypothesis)")
    print("="*70)

    for thresh in VIX_THRESHOLDS:
        for hold in HOLD_DAYS:
            print(f"  VIX>{thresh}, hold {hold}d ... ", end="", flush=True)
            result = evaluate_config(df, thresh, hold, "above")
            results["vix_above_results"].append(result)
            gates_passed = sum([
                result["gates"]["permutation"]["pass"],
                result["gates"]["regime_gap"]["pass"],
                result["gates"]["subperiod_consistency"]["pass"],
                result["gates"]["outlier_dependency"]["pass"]
            ])
            status = "PASS ALL" if result["all_gates_pass"] else f"FAIL ({gates_passed}/4)"
            print(f"Sharpe={result['sharpe']:.2f}, CAGR={result['cagr']:.1f}%, "
                  f"Exp={result['exposure_pct']:.0f}%, {status}")

    # Test VIX BELOW (buy when VIX is low — calm markets stay invested)
    print("\n" + "="*70)
    print("TESTING: Buy SPY when VIX BELOW threshold (calm=stay invested)")
    print("="*70)

    for thresh in VIX_THRESHOLDS:
        for hold in HOLD_DAYS:
            print(f"  VIX<{thresh}, hold {hold}d ... ", end="", flush=True)
            result = evaluate_config(df, thresh, hold, "below")
            results["vix_below_results"].append(result)
            gates_passed = sum([
                result["gates"]["permutation"]["pass"],
                result["gates"]["regime_gap"]["pass"],
                result["gates"]["subperiod_consistency"]["pass"],
                result["gates"]["outlier_dependency"]["pass"]
            ])
            status = "PASS ALL" if result["all_gates_pass"] else f"FAIL ({gates_passed}/4)"
            print(f"Sharpe={result['sharpe']:.2f}, CAGR={result['cagr']:.1f}%, "
                  f"Exp={result['exposure_pct']:.0f}%, {status}")

    # Summary
    above_passing = [r for r in results["vix_above_results"] if r["all_gates_pass"]]
    below_passing = [r for r in results["vix_below_results"] if r["all_gates_pass"]]

    best_above = max(results["vix_above_results"], key=lambda x: x["sharpe"]) if results["vix_above_results"] else None
    best_below = max(results["vix_below_results"], key=lambda x: x["sharpe"]) if results["vix_below_results"] else None

    results["summary"] = {
        "above_configs_tested": len(results["vix_above_results"]),
        "above_configs_passing_all_gates": len(above_passing),
        "below_configs_tested": len(results["vix_below_results"]),
        "below_configs_passing_all_gates": len(below_passing),
        "best_above": best_above,
        "best_below": best_below,
        "conclusion": ""
    }

    # Generate conclusion
    if len(above_passing) == 0 and len(below_passing) == 0:
        conclusion = ("NEITHER strategy passes all adversarial gates. The VIX threshold effect "
                     "is likely regime-dependent, inconsistent across sub-periods, or driven by outlier days.")
    elif len(above_passing) > 0 and len(below_passing) == 0:
        conclusion = (f"VIX-ABOVE (mean-reversion) has {len(above_passing)} configs passing all gates. "
                     "This suggests genuine mean-reversion edge after vol spikes.")
    elif len(below_passing) > 0 and len(above_passing) == 0:
        conclusion = (f"VIX-BELOW (calm markets) has {len(below_passing)} configs passing all gates. "
                     "Edge is from being invested in calm markets, not vol spike mean-reversion.")
    else:
        conclusion = (f"BOTH directions have passing configs (above: {len(above_passing)}, below: {len(below_passing)}). "
                     "Edge may be from market exposure itself, not VIX signal specifically.")

    results["summary"]["conclusion"] = conclusion

    # Print final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Buy & Hold SPY: Sharpe {results['buy_and_hold_spy']['sharpe']}")
    print(f"\nVIX ABOVE (vol spike → buy): {len(above_passing)}/{len(results['vix_above_results'])} pass all gates")
    if best_above:
        print(f"  Best: VIX>{best_above['vix_threshold']}, hold {best_above['hold_days']}d → "
              f"Sharpe {best_above['sharpe']}, CAGR {best_above['cagr']}%, Exposure {best_above['exposure_pct']}%")

    print(f"\nVIX BELOW (calm → stay in): {len(below_passing)}/{len(results['vix_below_results'])} pass all gates")
    if best_below:
        print(f"  Best: VIX<{best_below['vix_threshold']}, hold {best_below['hold_days']}d → "
              f"Sharpe {best_below['sharpe']}, CAGR {best_below['cagr']}%, Exposure {best_below['exposure_pct']}%")

    print(f"\nCONCLUSION: {conclusion}")

    # Save results
    output_path = OUTPUT_DIR / "results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
