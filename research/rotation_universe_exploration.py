"""
Rotation Universe Exploration
=============================
Compare multiple ETF universes using the same rotation logic:
- Momentum acceleration + RS rank change + anti-concentration
- 378d lookback, 21d hold period
- Walk-forward sliding (no lookahead)
- 5 bps transaction costs

Universes tested:
1. U.S. Sector ETFs (baseline) - K=3
2. Country/Region ETFs - K=4
3. Asset Class / Factor ETFs - K=4
4. Thematic/Industry ETFs - K=4
5. Wide Sector + International Combo - K=4
6. Leveraged Sector Rotation (inverse-vol sized, target 15% ann vol) - K=3

Output: /home/jupiter/Lvl3Quant/output/rotation_universe_exploration/
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/rotation_universe_exploration")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# UNIVERSE DEFINITIONS
# ============================================================

UNIVERSES = {
    "U.S. Sectors (K=3)": {
        "tickers": ["XLK", "XLV", "XLE", "XLF", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY"],
        "K": 3,
        "vol_target": None,
    },
    "Country/Region (K=4)": {
        "tickers": ["EWJ", "EWG", "EWU", "EWA", "EWC", "EWZ", "EWY", "EWT", "INDA", "EWS",
                    "FXI", "EZU", "VGK", "EEM", "VWO", "EFA"],
        "K": 4,
        "vol_target": None,
    },
    "Asset Class/Factor (K=4)": {
        "tickers": ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "SLV", "DBC",
                    "VNQ", "HYG", "LQD", "UUP", "XLU"],
        "K": 4,
        "vol_target": None,
    },
    "Thematic/Industry (K=4)": {
        "tickers": ["XBI", "ARKK", "SMH", "XHB", "XOP", "KRE", "IBB", "ITA", "HACK", "TAN",
                    "JETS", "XME", "ITB", "CIBR", "SOXX"],
        "K": 4,
        "vol_target": None,
    },
    "Wide Sector+Intl (K=4)": {
        "tickers": ["XLK", "XLV", "XLE", "XLF", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY",
                    "EFA", "EEM", "GLD", "TLT", "VNQ"],
        "K": 4,
        "vol_target": None,
    },
    "Leveraged Sector (Vol-Target 15%)": {
        "tickers": ["XLK", "XLV", "XLE", "XLF", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY"],
        "K": 3,
        "vol_target": 0.15,
    },
}

# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data(all_tickers, start="2015-01-01", end="2026-07-11"):
    """Download adjusted close prices for all tickers."""
    print(f"Downloading data for {len(all_tickers)} unique tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data
    # Drop tickers with insufficient data
    prices = prices.dropna(axis=1, how='all')
    print(f"  Got data for {prices.shape[1]} tickers, {prices.shape[0]} trading days")
    print(f"  Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    return prices


# ============================================================
# ROTATION STRATEGY LOGIC
# ============================================================

def compute_momentum_acceleration(prices, lookback=378):
    """
    Momentum acceleration: rate of change of momentum.
    Uses 378d lookback split into two halves to measure acceleration.
    """
    half = lookback // 2
    mom_full = prices.pct_change(lookback)
    mom_half = prices.pct_change(half)
    # Acceleration = recent momentum - older momentum (normalized)
    acceleration = mom_half - (mom_full - mom_half)
    return acceleration


def compute_rs_rank_change(prices, lookback=378, rank_window=63):
    """
    Relative strength rank change over rank_window days.
    Positive rank change = improving relative strength.
    """
    mom = prices.pct_change(lookback)
    ranks = mom.rank(axis=1, pct=True)
    rank_change = ranks - ranks.shift(rank_window)
    return rank_change


def compute_composite_score(prices, lookback=378):
    """
    Composite rotation score: momentum acceleration + RS rank change.
    Both normalized cross-sectionally.
    """
    accel = compute_momentum_acceleration(prices, lookback)
    rs_change = compute_rs_rank_change(prices, lookback)

    # Cross-sectional z-score normalization
    accel_z = accel.sub(accel.mean(axis=1), axis=0).div(accel.std(axis=1), axis=0)
    rs_z = rs_change.sub(rs_change.mean(axis=1), axis=0).div(rs_change.std(axis=1), axis=0)

    composite = 0.5 * accel_z + 0.5 * rs_z
    return composite


def anti_concentration_filter(scores, max_weight=0.40, K=3):
    """
    Select top K assets but cap any single position at max_weight.
    Returns equal weight among selected (with cap redistribution).
    """
    # Simple equal weight among top K (anti-concentration by diversity)
    # The cap ensures no single asset > 40% (matters if K < 3)
    n_assets = scores.shape[1] if hasattr(scores, 'shape') else len(scores)
    weight = 1.0 / K
    if weight > max_weight:
        weight = max_weight
    return weight


def run_rotation_strategy(prices, tickers, K=3, lookback=378, hold_period=21,
                          cost_bps=5, vol_target=None):
    """
    Walk-forward rotation strategy:
    - Every hold_period days, score universe and select top K
    - Equal weight among top K (with anti-concentration)
    - Apply transaction costs
    - Optionally scale by inverse volatility to target vol
    """
    # Filter to available tickers
    available = [t for t in tickers if t in prices.columns]
    if len(available) < K + 1:
        return None, f"Only {len(available)} tickers available, need at least {K+1}"

    p = prices[available].copy()
    # Need lookback + some buffer
    min_data = lookback + 63  # extra 63 for rank change window

    # Find first valid date (all assets have enough data)
    first_valid = p.dropna().index[0] if not p.dropna().empty else None
    # Actually find where we have enough history
    valid_start_idx = min_data
    if valid_start_idx >= len(p):
        return None, "Insufficient data for lookback period"

    # Compute scores
    scores = compute_composite_score(p, lookback)

    # Walk-forward: rebalance every hold_period days
    returns = p.pct_change()
    portfolio_returns = []
    rebalance_dates = []
    holdings_history = []
    turnover_list = []

    current_holdings = {}  # ticker -> weight
    rebalance_idx = valid_start_idx

    dates = p.index[valid_start_idx:]

    i = 0
    for date in dates:
        idx_in_full = p.index.get_loc(date)

        if i == 0 or i % hold_period == 0:
            # Rebalance
            day_scores = scores.loc[date]
            valid_scores = day_scores.dropna()

            if len(valid_scores) < K:
                # Not enough valid scores, hold cash
                new_holdings = {}
            else:
                # Select top K
                top_k = valid_scores.nlargest(K).index.tolist()
                weight = 1.0 / K
                # Anti-concentration cap at 40%
                weight = min(weight, 0.40)
                new_holdings = {t: weight for t in top_k}
                # Redistribute if capped
                total = sum(new_holdings.values())
                if total < 1.0 and len(new_holdings) > 0:
                    # Spread remainder
                    remainder = 1.0 - total
                    for t in new_holdings:
                        new_holdings[t] += remainder / len(new_holdings)

            # Compute turnover
            all_tickers_held = set(list(current_holdings.keys()) + list(new_holdings.keys()))
            turnover = sum(abs(new_holdings.get(t, 0) - current_holdings.get(t, 0))
                         for t in all_tickers_held) / 2.0
            turnover_list.append(turnover)
            rebalance_dates.append(date)
            current_holdings = new_holdings
            holdings_history.append(current_holdings.copy())

        # Compute daily return
        if current_holdings:
            day_ret = sum(current_holdings.get(t, 0) * returns.loc[date, t]
                         for t in current_holdings if t in returns.columns and not pd.isna(returns.loc[date, t]))
            # Apply cost on rebalance days
            if i == 0 or i % hold_period == 0:
                day_ret -= turnover_list[-1] * (cost_bps / 10000.0) * 2  # round-trip
        else:
            day_ret = 0.0

        portfolio_returns.append({"date": date, "return": day_ret})
        i += 1

    df_ret = pd.DataFrame(portfolio_returns).set_index("date")
    df_ret.index = pd.to_datetime(df_ret.index)

    # Apply vol targeting if specified
    if vol_target is not None:
        # Rolling 63-day realized vol, scale position size
        rolling_vol = df_ret["return"].rolling(63).std() * np.sqrt(252)
        rolling_vol = rolling_vol.clip(lower=0.05)  # floor at 5%
        scale = vol_target / rolling_vol
        scale = scale.clip(upper=2.0)  # cap leverage at 2x
        df_ret["return"] = df_ret["return"] * scale.shift(1)  # use lagged vol (no lookahead)

    result = {
        "returns": df_ret,
        "rebalance_dates": rebalance_dates,
        "turnover_list": turnover_list,
        "available_tickers": available,
        "start_date": df_ret.index[0],
        "n_rebalances": len(rebalance_dates),
    }
    return result, None


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def compute_metrics(returns_series, spy_returns=None):
    """Compute comprehensive performance metrics."""
    r = returns_series.dropna()
    if len(r) < 252:
        return None

    # Annualized metrics
    n_years = len(r) / 252.0
    cum_ret = (1 + r).prod() - 1
    cagr = (1 + cum_ret) ** (1.0 / n_years) - 1

    ann_vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / (r.std() * np.sqrt(252)) if r.std() > 0 else 0

    # Sortino
    downside = r[r < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
    sortino = (r.mean() * 252) / downside_vol

    # Max Drawdown
    cum = (1 + r).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Worst year
    yearly = r.groupby(r.index.year).sum()
    worst_year = yearly.min()
    worst_year_label = yearly.idxmin()

    # Hit rate vs universe median (approximated by positive return days)
    hit_rate = (r > 0).mean()

    # Correlation to SPY
    spy_corr = np.nan
    if spy_returns is not None:
        common = r.index.intersection(spy_returns.index)
        if len(common) > 100:
            spy_corr = r.loc[common].corr(spy_returns.loc[common])

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDD": max_dd,
        "Calmar": calmar,
        "AnnVol": ann_vol,
        "HitRate": hit_rate,
        "SPY_Corr": spy_corr,
        "WorstYear": worst_year,
        "WorstYearLabel": worst_year_label,
        "TotalReturn": cum_ret,
        "N_Years": n_years,
    }


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(prices, tickers, K, lookback, hold_period, cost_bps,
                     vol_target, actual_sharpe, n_permutations=500):
    """
    Permutation test: shuffle the cross-sectional scores each rebalance
    to see if selection skill is real.
    """
    available = [t for t in tickers if t in prices.columns]
    p = prices[available].copy()
    min_data = lookback + 63
    if min_data >= len(p):
        return None

    returns = p.pct_change()
    valid_start_idx = min_data
    dates = p.index[valid_start_idx:]

    # Pre-compute scores
    scores = compute_composite_score(p, lookback)

    random_sharpes = []
    rng = np.random.default_rng(42)

    for perm in range(n_permutations):
        portfolio_returns = []
        current_holdings = {}
        i = 0

        for date in dates:
            if i == 0 or i % hold_period == 0:
                day_scores = scores.loc[date].dropna()
                if len(day_scores) >= K:
                    # SHUFFLE scores randomly
                    shuffled = day_scores.copy()
                    shuffled_vals = shuffled.values.copy()
                    rng.shuffle(shuffled_vals)
                    shuffled = pd.Series(shuffled_vals, index=shuffled.index)
                    top_k = shuffled.nlargest(K).index.tolist()
                    weight = 1.0 / K
                    current_holdings = {t: weight for t in top_k}
                else:
                    current_holdings = {}

            if current_holdings:
                day_ret = sum(current_holdings.get(t, 0) * returns.loc[date, t]
                             for t in current_holdings
                             if t in returns.columns and not pd.isna(returns.loc[date, t]))
            else:
                day_ret = 0.0

            portfolio_returns.append(day_ret)
            i += 1

        pr = pd.Series(portfolio_returns)
        if pr.std() > 0:
            random_sharpes.append((pr.mean() * 252) / (pr.std() * np.sqrt(252)))
        else:
            random_sharpes.append(0)

        if (perm + 1) % 100 == 0:
            print(f"    Permutation {perm+1}/{n_permutations}")

    random_sharpes = np.array(random_sharpes)
    p_value = (random_sharpes >= actual_sharpe).mean()

    return {
        "actual_sharpe": actual_sharpe,
        "mean_random_sharpe": random_sharpes.mean(),
        "std_random_sharpe": random_sharpes.std(),
        "p_value": p_value,
        "percentile": (random_sharpes < actual_sharpe).mean() * 100,
        "n_permutations": n_permutations,
    }


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("=" * 80)
    print("ROTATION UNIVERSE EXPLORATION")
    print("=" * 80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Lookback: 378d, Hold: 21d, Cost: 5 bps")
    print()

    # Collect all unique tickers
    all_tickers = set()
    for name, cfg in UNIVERSES.items():
        all_tickers.update(cfg["tickers"])
    all_tickers.add("SPY")  # For correlation benchmark
    all_tickers = sorted(list(all_tickers))

    # Download data
    prices = download_data(all_tickers, start="2015-01-01", end="2026-07-11")

    # SPY returns for correlation
    spy_returns = prices["SPY"].pct_change().dropna() if "SPY" in prices.columns else None

    # Run each universe
    results = {}
    all_metrics = {}

    print("\n" + "=" * 80)
    print("RUNNING ROTATION STRATEGIES")
    print("=" * 80)

    for name, cfg in UNIVERSES.items():
        print(f"\n{'─' * 60}")
        print(f"Universe: {name}")
        print(f"  Tickers: {cfg['tickers']}")
        print(f"  K={cfg['K']}, vol_target={cfg['vol_target']}")

        result, error = run_rotation_strategy(
            prices, cfg["tickers"], K=cfg["K"],
            lookback=378, hold_period=21, cost_bps=5,
            vol_target=cfg["vol_target"]
        )

        if error:
            print(f"  ERROR: {error}")
            continue

        print(f"  Start: {result['start_date'].strftime('%Y-%m-%d')}")
        print(f"  Rebalances: {result['n_rebalances']}")
        print(f"  Avg Turnover: {np.mean(result['turnover_list']):.2%}")

        # Compute metrics
        metrics = compute_metrics(result["returns"]["return"], spy_returns)
        if metrics is None:
            print(f"  SKIP: insufficient data for metrics")
            continue

        metrics["AvgTurnover"] = np.mean(result["turnover_list"])
        metrics["N_Rebalances"] = result["n_rebalances"]
        metrics["StartDate"] = result["start_date"].strftime("%Y-%m-%d")
        metrics["N_Tickers"] = len(result["available_tickers"])

        results[name] = result
        all_metrics[name] = metrics

        print(f"  Sharpe: {metrics['Sharpe']:.3f}")
        print(f"  CAGR: {metrics['CAGR']:.2%}")
        print(f"  MaxDD: {metrics['MaxDD']:.2%}")
        print(f"  Sortino: {metrics['Sortino']:.3f}")

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n\n" + "=" * 80)
    print("COMPARISON TABLE (sorted by Sharpe)")
    print("=" * 80)

    df_metrics = pd.DataFrame(all_metrics).T
    df_metrics = df_metrics.sort_values("Sharpe", ascending=False)

    # Format for display
    display_cols = ["Sharpe", "Sortino", "CAGR", "MaxDD", "Calmar", "SPY_Corr",
                    "HitRate", "AvgTurnover", "WorstYear", "N_Rebalances", "StartDate"]

    print(f"\n{'Universe':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} "
          f"{'Calmar':>7} {'SPY_r':>6} {'HitR':>5} {'Turn':>5} {'WrstYr':>7} {'#Reb':>5} {'Start':>11}")
    print("─" * 130)

    for name, row in df_metrics.iterrows():
        print(f"{name:<35} {row['Sharpe']:>7.3f} {row['Sortino']:>8.3f} "
              f"{row['CAGR']:>6.1%} {row['MaxDD']:>6.1%} {row['Calmar']:>7.3f} "
              f"{row['SPY_Corr']:>6.2f} {row['HitRate']:>4.1%} {row['AvgTurnover']:>4.1%} "
              f"{row['WorstYear']:>6.1%} {row['N_Rebalances']:>5.0f} {row['StartDate']:>11}")

    # ============================================================
    # PERMUTATION TESTS ON TOP 2
    # ============================================================
    print("\n\n" + "=" * 80)
    print("PERMUTATION TESTS (top 2 universes, 500 iterations)")
    print("=" * 80)

    top2 = df_metrics.index[:2].tolist()

    perm_results = {}
    for name in top2:
        print(f"\n  Testing: {name}")
        cfg = UNIVERSES[name]
        actual_sharpe = all_metrics[name]["Sharpe"]

        perm = permutation_test(
            prices, cfg["tickers"], K=cfg["K"],
            lookback=378, hold_period=21, cost_bps=5,
            vol_target=cfg["vol_target"],
            actual_sharpe=actual_sharpe, n_permutations=500
        )

        if perm:
            perm_results[name] = perm
            print(f"    Actual Sharpe: {perm['actual_sharpe']:.3f}")
            print(f"    Random Mean:   {perm['mean_random_sharpe']:.3f} +/- {perm['std_random_sharpe']:.3f}")
            print(f"    p-value:       {perm['p_value']:.4f}")
            print(f"    Percentile:    {perm['percentile']:.1f}%")
            if perm['p_value'] < 0.05:
                print(f"    SIGNIFICANT at 5% level")
            else:
                print(f"    NOT significant at 5% level")

    # ============================================================
    # SAVE OUTPUTS
    # ============================================================
    print("\n\n" + "=" * 80)
    print("SAVING OUTPUTS")
    print("=" * 80)

    # Save metrics table
    df_metrics.to_csv(OUTPUT_DIR / "comparison_table.csv")
    print(f"  Saved: comparison_table.csv")

    # Save daily returns for each universe
    for name, result in results.items():
        safe_name = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        result["returns"].to_csv(OUTPUT_DIR / f"returns_{safe_name}.csv")

    # Save permutation results
    perm_save = {}
    for name, perm in perm_results.items():
        perm_save[name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                          for k, v in perm.items()}
    with open(OUTPUT_DIR / "permutation_results.json", "w") as f:
        json.dump(perm_save, f, indent=2)
    print(f"  Saved: permutation_results.json")

    # Save full summary
    summary = {
        "run_date": datetime.now().isoformat(),
        "parameters": {"lookback": 378, "hold_period": 21, "cost_bps": 5},
        "metrics": {},
    }
    for name, m in all_metrics.items():
        summary["metrics"][name] = {k: (float(v) if isinstance(v, (np.floating, np.integer, float)) else str(v))
                                    for k, v in m.items()}
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: summary.json")

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nBest Sharpe:  {df_metrics.index[0]} ({df_metrics.iloc[0]['Sharpe']:.3f})")
    print(f"Best Sortino: {df_metrics.sort_values('Sortino', ascending=False).index[0]} "
          f"({df_metrics['Sortino'].max():.3f})")
    print(f"Best CAGR:    {df_metrics.sort_values('CAGR', ascending=False).index[0]} "
          f"({df_metrics['CAGR'].max():.2%})")
    print(f"Lowest MaxDD: {df_metrics.sort_values('MaxDD', ascending=False).index[0]} "
          f"({df_metrics['MaxDD'].max():.2%})")
    print(f"Lowest SPY Corr: {df_metrics.sort_values('SPY_Corr').index[0]} "
          f"({df_metrics['SPY_Corr'].min():.2f})")

    print(f"\nKey findings:")
    baseline_sharpe = all_metrics.get("U.S. Sectors (K=3)", {}).get("Sharpe", 0)
    for name in df_metrics.index:
        if name != "U.S. Sectors (K=3)":
            s = all_metrics[name]["Sharpe"]
            diff = s - baseline_sharpe
            print(f"  {name}: Sharpe {s:.3f} ({'+' if diff > 0 else ''}{diff:.3f} vs baseline)")

    print(f"\nPermutation test results:")
    for name, perm in perm_results.items():
        sig = "SIGNIFICANT" if perm["p_value"] < 0.05 else "NOT significant"
        print(f"  {name}: p={perm['p_value']:.4f} ({sig})")

    print(f"\n{'=' * 80}")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
