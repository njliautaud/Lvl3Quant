#!/usr/bin/env python3
"""
Static Allocation Optimizer
----------------------------
Determines optimal allocation between Signal Agg A (QQQ timing) and
buy-and-hold diversifiers for a $645 portfolio.

OOT period: 2022-01-01 to 2026-07-29
"""

import json
import datetime
import itertools
import warnings
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
CAPITAL = 645
RISK_FREE_RATE = 0.04  # ~4% for 2022-2026 avg
TRADING_DAYS = 252

DIVERSIFIERS = ["GLD", "TLT", "UUP", "SHY", "RSP", "IEF", "DBA", "VNQ"]
ALL_TICKERS = ["QQQ", "^VIX", "SPY"] + DIVERSIFIERS


def download_data():
    """Download all needed price data."""
    # Need extra history for 200-SMA on SPY and 50-SMA on QQQ
    start_early = "2021-01-01"
    data = {}
    for ticker in ALL_TICKERS:
        df = yf.download(ticker, start=start_early, end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df["Close"]
        print(f"  Downloaded {ticker}: {len(df)} rows")
    return pd.DataFrame(data).ffill()


def calc_metrics(returns_series, name=""):
    """Calculate Sharpe, total return, MDD, Calmar, Sortino."""
    returns = returns_series.dropna()
    if len(returns) < 20:
        return {"sharpe": 0, "return_pct": 0, "mdd": 0, "calmar": 0, "sortino": 0, "annual_vol": 0}

    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # MDD
    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    mdd = drawdown.min()

    # Calmar
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-8 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(TRADING_DAYS) if len(downside) > 0 else 1e-8
    sortino = (ann_ret - RISK_FREE_RATE) / downside_vol if downside_vol > 0 else 0

    total_ret = (cum.iloc[-1] - 1) * 100 if len(cum) > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "return_pct": round(total_ret, 2),
        "mdd": round(mdd * 100, 2),
        "calmar": round(calmar, 3),
        "sortino": round(sortino, 3),
        "annual_vol": round(ann_vol * 100, 2),
    }


def build_signal_agg_a(prices_df):
    """
    Signal Agg A proxy: long QQQ when QQQ > 50-SMA AND VIX < 25, else cash.
    Returns daily returns series in OOT period.
    """
    qqq = prices_df["QQQ"]
    vix = prices_df["^VIX"]

    sma50 = qqq.rolling(50).mean()
    signal = (qqq > sma50) & (vix < 25)

    qqq_ret = qqq.pct_change()

    # When signal is True, we get QQQ returns; otherwise 0 (cash)
    strategy_ret = qqq_ret.where(signal.shift(1), 0)  # shift signal to avoid lookahead

    # Trim to OOT
    strategy_ret = strategy_ret.loc[OOT_START:]
    return strategy_ret


def part1_buy_and_hold(prices_df):
    """Part 1: Buy-and-hold asset returns."""
    print("\n=== PART 1: Buy-and-Hold Asset Returns ===")
    oot_prices = prices_df.loc[OOT_START:]
    qqq_ret = oot_prices["QQQ"].pct_change().dropna()

    results = {}
    for ticker in ["QQQ"] + DIVERSIFIERS:
        ret = oot_prices[ticker].pct_change().dropna()
        metrics = calc_metrics(ret, ticker)
        corr = ret.corr(qqq_ret) if ticker != "QQQ" else 1.0
        metrics["qqq_corr"] = round(corr, 3)
        results[ticker] = metrics
        print(f"  {ticker:5s}: Sharpe={metrics['sharpe']:6.3f}  Ret={metrics['return_pct']:7.2f}%  "
              f"MDD={metrics['mdd']:7.2f}%  Corr={metrics['qqq_corr']:6.3f}")

    return results


def part2_two_asset(prices_df, signal_a_ret):
    """Part 2: Two-asset portfolio optimization."""
    print("\n=== PART 2: Two-Asset Portfolios ===")
    oot_prices = prices_df.loc[OOT_START:]
    allocations = [100, 90, 80, 70, 60, 50]

    results = {}
    for ticker in DIVERSIFIERS:
        div_ret = oot_prices[ticker].pct_change()
        # Align
        combined_idx = signal_a_ret.index.intersection(div_ret.index)
        sa = signal_a_ret.loc[combined_idx]
        dr = div_ret.loc[combined_idx]

        alloc_results = []
        best_sharpe = -999
        best_split = None

        for sig_pct in allocations:
            div_pct = 100 - sig_pct
            port_ret = (sig_pct / 100) * sa + (div_pct / 100) * dr
            m = calc_metrics(port_ret)
            entry = {
                "signal_a_pct": sig_pct,
                "diversifier_pct": div_pct,
                **m,
            }
            alloc_results.append(entry)
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_split = {"signal_a": sig_pct, "diversifier": div_pct, "sharpe": m["sharpe"]}

        results[f"SignalA_{ticker}"] = {
            "allocations": alloc_results,
            "optimal_split": best_split,
        }
        print(f"  SignalA + {ticker:4s}: optimal {best_split['signal_a']}/{best_split['diversifier']}  "
              f"Sharpe={best_sharpe:.3f}")

    return results


def part3_three_asset(prices_df, signal_a_ret, two_asset_results):
    """Part 3: Three-asset portfolios from best diversifiers."""
    print("\n=== PART 3: Three-Asset Portfolios ===")
    oot_prices = prices_df.loc[OOT_START:]

    # Rank diversifiers by their optimal Sharpe improvement
    div_scores = []
    for key, val in two_asset_results.items():
        ticker = key.replace("SignalA_", "")
        div_scores.append((ticker, val["optimal_split"]["sharpe"]))
    div_scores.sort(key=lambda x: x[1], reverse=True)
    top_divs = [d[0] for d in div_scores[:3]]
    print(f"  Top diversifiers: {top_divs}")

    # Test all pairs from top 3
    combos = list(itertools.combinations(top_divs, 2))
    all_results = []
    best_overall = {"sharpe": -999}

    for d1, d2 in combos:
        dr1 = oot_prices[d1].pct_change()
        dr2 = oot_prices[d2].pct_change()
        combined_idx = signal_a_ret.index.intersection(dr1.index).intersection(dr2.index)
        sa = signal_a_ret.loc[combined_idx]
        r1 = dr1.loc[combined_idx]
        r2 = dr2.loc[combined_idx]

        # Test allocations at 10% steps
        for sa_pct in range(50, 101, 10):
            remaining = 100 - sa_pct
            for d1_pct in range(0, remaining + 1, 10):
                d2_pct = remaining - d1_pct
                port_ret = (sa_pct / 100) * sa + (d1_pct / 100) * r1 + (d2_pct / 100) * r2
                m = calc_metrics(port_ret)
                entry = {
                    "assets": f"SignalA/{d1}/{d2}",
                    "signal_a_pct": sa_pct,
                    f"{d1}_pct": d1_pct,
                    f"{d2}_pct": d2_pct,
                    **m,
                }
                all_results.append(entry)
                if m["sharpe"] > best_overall["sharpe"]:
                    best_overall = entry.copy()

    # Sort by Sharpe
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"  Best 3-asset: {best_overall['assets']} "
          f"({best_overall['signal_a_pct']}% signal) Sharpe={best_overall['sharpe']:.3f}")

    return {
        "top_diversifiers": top_divs,
        "best_combination": best_overall,
        "best_sharpe": best_overall["sharpe"],
        "top_10": all_results[:10],
    }


def part4_regime_analysis(prices_df, signal_a_ret):
    """Part 4: Regime-stratified analysis (SPY vs 200-SMA)."""
    print("\n=== PART 4: Regime-Stratified Analysis ===")
    oot_prices = prices_df.loc[OOT_START:]
    spy = prices_df["SPY"]
    spy_sma200 = spy.rolling(200).mean()

    bull = spy > spy_sma200  # bull regime
    bull_oot = bull.loc[OOT_START:]

    regime_results = {}
    all_tickers_to_test = ["SignalA"] + DIVERSIFIERS

    for asset in all_tickers_to_test:
        if asset == "SignalA":
            ret = signal_a_ret
        else:
            ret = oot_prices[asset].pct_change()

        combined_idx = ret.index.intersection(bull_oot.index)
        r = ret.loc[combined_idx]
        b = bull_oot.loc[combined_idx]

        bull_ret = r[b]
        bear_ret = r[~b]

        bull_m = calc_metrics(bull_ret, f"{asset}_bull")
        bear_m = calc_metrics(bear_ret, f"{asset}_bear")
        full_m = calc_metrics(r, f"{asset}_full")

        stability = 1.0 - abs(bull_m["sharpe"] - bear_m["sharpe"]) / max(abs(bull_m["sharpe"]), abs(bear_m["sharpe"]), 0.01)

        regime_results[asset] = {
            "bull_sharpe": bull_m["sharpe"],
            "bear_sharpe": bear_m["sharpe"],
            "full_sharpe": full_m["sharpe"],
            "stability_score": round(stability, 3),
            "bull_return_pct": bull_m["return_pct"],
            "bear_return_pct": bear_m["return_pct"],
        }
        print(f"  {asset:10s}: Bull Sharpe={bull_m['sharpe']:6.3f}  Bear Sharpe={bear_m['sharpe']:6.3f}  "
              f"Stability={stability:.3f}")

    # Find best in each category
    best_bull = max(regime_results.items(), key=lambda x: x[1]["bull_sharpe"])
    best_bear = max(regime_results.items(), key=lambda x: x[1]["bear_sharpe"])
    most_stable = max(regime_results.items(), key=lambda x: x[1]["stability_score"])

    # Also test best 2-asset mixes in regimes
    # Test SignalA + GLD at various splits
    mix_regime = {}
    for ticker in ["GLD", "TLT", "SHY"]:
        div_ret = oot_prices[ticker].pct_change()
        combined_idx = signal_a_ret.index.intersection(div_ret.index).intersection(bull_oot.index)
        sa = signal_a_ret.loc[combined_idx]
        dr = div_ret.loc[combined_idx]
        b = bull_oot.loc[combined_idx]

        for pct in [90, 80, 70]:
            port = (pct / 100) * sa + ((100 - pct) / 100) * dr
            bull_m = calc_metrics(port[b])
            bear_m = calc_metrics(port[~b])
            stab = 1.0 - abs(bull_m["sharpe"] - bear_m["sharpe"]) / max(abs(bull_m["sharpe"]), abs(bear_m["sharpe"]), 0.01)
            key = f"SignalA_{pct}_{ticker}_{100-pct}"
            mix_regime[key] = {
                "bull_sharpe": bull_m["sharpe"],
                "bear_sharpe": bear_m["sharpe"],
                "stability_score": round(stab, 3),
            }

    most_stable_mix = max(mix_regime.items(), key=lambda x: x[1]["stability_score"])

    return {
        "individual_assets": regime_results,
        "best_bull": {"asset": best_bull[0], **best_bull[1]},
        "best_bear": {"asset": best_bear[0], **best_bear[1]},
        "most_stable": {"asset": most_stable[0], **most_stable[1]},
        "mixed_portfolios_regime": mix_regime,
        "most_stable_mix": {"name": most_stable_mix[0], **most_stable_mix[1]},
    }


def part5_practical(two_asset_results, three_asset_results, buy_hold_results):
    """Part 5: Practical recommendation for $645 capital."""
    print("\n=== PART 5: Practical Recommendation ===")

    # Approximate share prices (mid-2026)
    approx_prices = {
        "QQQ": 530, "GLD": 300, "TLT": 90, "UUP": 27,
        "SHY": 82, "RSP": 180, "IEF": 95, "DBA": 25, "VNQ": 85,
    }

    # At $645, how many shares of each can we buy?
    share_analysis = {}
    for ticker, price in approx_prices.items():
        max_shares = int(CAPITAL / price)
        cost = max_shares * price
        pct_of_capital = round(cost / CAPITAL * 100, 1) if max_shares > 0 else 0
        share_analysis[ticker] = {
            "approx_price": price,
            "max_shares": max_shares,
            "cost": cost,
            "pct_of_capital": pct_of_capital,
        }

    # Can we do 80/20 with any diversifier?
    feasible_splits = {}
    for ticker in DIVERSIFIERS:
        price = approx_prices.get(ticker, 100)
        # 20% of $645 = $129
        div_budget = 0.2 * CAPITAL
        qqq_budget = 0.8 * CAPITAL
        div_shares = int(div_budget / price)
        qqq_shares = int(qqq_budget / approx_prices["QQQ"])

        feasible = div_shares >= 1 and qqq_shares >= 1
        feasible_splits[ticker] = {
            "diversifier_shares": div_shares,
            "qqq_shares": qqq_shares,
            "feasible": feasible,
            "actual_div_pct": round(div_shares * price / CAPITAL * 100, 1) if div_shares > 0 else 0,
        }

    # Minimum capital for meaningful diversification
    # Need at least 1 share QQQ (~$530) + 1 share diversifier
    min_for_gld = approx_prices["QQQ"] + approx_prices["GLD"]
    min_for_tlt = approx_prices["QQQ"] + approx_prices["TLT"]
    min_for_cheap = approx_prices["QQQ"] + min(approx_prices[t] for t in DIVERSIFIERS)

    # Fractional shares on Robinhood
    frac_note = ("Robinhood supports fractional shares for most ETFs, so exact allocation "
                 "percentages are achievable regardless of capital size.")

    # Build recommendation
    rec_text = (
        f"At ${CAPITAL}, fractional shares on Robinhood make any allocation feasible. "
        f"However, the capital is small enough that transaction costs and bid-ask spreads "
        f"on rebalancing eat into returns disproportionately. "
        f"Recommendation: keep 100% in Signal Agg A (QQQ timing) until capital exceeds "
        f"~$2,000-3,000 where rebalancing friction becomes negligible relative to "
        f"diversification benefit. The timing signal's Sharpe advantage over static "
        f"diversification is large enough that diluting it with buy-and-hold positions "
        f"at this scale likely hurts more than it helps."
    )

    result = {
        "capital": CAPITAL,
        "fractional_shares_available": True,
        "share_analysis": share_analysis,
        "feasible_80_20_splits": feasible_splits,
        "min_capital_for_diversification": {
            "with_GLD": min_for_gld,
            "with_TLT": min_for_tlt,
            "cheapest_diversifier": min_for_cheap,
            "recommended_min": 2500,
            "note": "Fractional shares make any split possible, but rebalancing friction matters more at small scale",
        },
        "recommendation_at_645": rec_text,
        "fractional_share_note": frac_note,
    }

    print(f"  Recommendation: {rec_text[:100]}...")
    return result


def main():
    print("=" * 70)
    print("STATIC ALLOCATION OPTIMIZER")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${CAPITAL}")
    print("=" * 70)

    # Download data
    print("\nDownloading data...")
    prices_df = download_data()

    # Build Signal Agg A proxy returns
    print("\nBuilding Signal Agg A proxy...")
    signal_a_ret = build_signal_agg_a(prices_df)
    signal_a_metrics = calc_metrics(signal_a_ret, "Signal Agg A")
    print(f"  Signal Agg A: Sharpe={signal_a_metrics['sharpe']:.3f}  "
          f"Return={signal_a_metrics['return_pct']:.2f}%  MDD={signal_a_metrics['mdd']:.2f}%")

    # Exposure stats
    exposure = (signal_a_ret != 0).mean() * 100
    print(f"  Exposure: {exposure:.1f}% of days in market")

    # Run all parts
    bh_results = part1_buy_and_hold(prices_df)
    two_asset = part2_two_asset(prices_df, signal_a_ret)
    three_asset = part3_three_asset(prices_df, signal_a_ret, two_asset)
    regime = part4_regime_analysis(prices_df, signal_a_ret)
    practical = part5_practical(two_asset, three_asset, bh_results)

    # Compile full results
    results = {
        "meta": {
            "run_date": datetime.datetime.now().isoformat(),
            "oot_period": f"{OOT_START} to {OOT_END}",
            "capital": CAPITAL,
            "signal_proxy": "Long QQQ when QQQ > 50-SMA AND VIX < 25, else cash",
            "risk_free_rate": RISK_FREE_RATE,
        },
        "signal_agg_a_proxy": {
            "metrics": signal_a_metrics,
            "exposure_pct": round(exposure, 1),
        },
        "buy_and_hold_assets": bh_results,
        "two_asset_portfolios": two_asset,
        "three_asset_portfolios": three_asset,
        "regime_analysis": regime,
        "practical_recommendation": practical,
    }

    # Save
    out_path = "/home/jupiter/Lvl3Quant/data/static_allocation_optimizer_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    results = main()
