#!/usr/bin/env python3
"""
Optimal Allocation Backtest: Signal Agg A (QQQ proxy) vs GLD+UUP Vol-Targeted Diversifier
Finds efficient frontier for Robinhood account allocation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ─── CONFIG ───
START = "2022-01-01"
END = "2026-07-29"
TICKERS = ["QQQ", "GLD", "UUP"]
VOL_TARGET = 0.08  # 8% annualized
VOL_LOOKBACK = 20  # 20-day rolling vol
SLIPPAGE_BPS = 2  # 0.02%
RISK_FREE = 0.045  # approximate avg risk-free over period
TRADING_DAYS = 252
SPLITS = [round(x / 10, 1) for x in range(11)]  # 0.0 to 1.0 in 0.1 steps
REBAL_FREQS = ["monthly", "quarterly"]

OUTPUT_JSON = Path("/home/jupiter/Lvl3Quant/data/optimal_allocation_results.json")


def download_data():
    """Download price data for QQQ, GLD, UUP."""
    print(f"Downloading {TICKERS} from {START} to {END}...")
    data = yf.download(TICKERS, start=START, end=END, auto_adjust=True, progress=False)
    # yfinance returns multi-level columns for multiple tickers
    closes = data["Close"]
    closes = closes.dropna()
    print(f"  Got {len(closes)} trading days from {closes.index[0].date()} to {closes.index[-1].date()}")
    return closes


def compute_gld_uup_vol_targeted(closes: pd.DataFrame) -> pd.Series:
    """
    GLD+UUP equal-weight, vol-targeted to 8% annualized, weekly rebalance (Friday).
    Returns daily returns series.
    """
    gld_ret = closes["GLD"].pct_change()
    uup_ret = closes["UUP"].pct_change()

    # Equal-weight portfolio returns (before vol targeting)
    ew_ret = 0.5 * gld_ret + 0.5 * uup_ret

    # 20-day rolling vol of the equal-weight portfolio
    rolling_vol = ew_ret.rolling(VOL_LOOKBACK).std() * np.sqrt(TRADING_DAYS)

    # Vol-target scalar: target_vol / realized_vol, capped at 2x leverage
    vol_scalar = (VOL_TARGET / rolling_vol).clip(upper=2.0)

    # Weekly rebalance: only update scalar on Fridays
    # On non-Friday days, carry forward the last Friday's scalar
    is_friday = closes.index.dayofweek == 4
    scalar_series = vol_scalar.copy()
    scalar_series[~is_friday] = np.nan
    scalar_series = scalar_series.ffill()
    scalar_series = scalar_series.fillna(1.0)  # before first Friday

    # Apply vol targeting + slippage on rebalance days
    slippage = pd.Series(0.0, index=closes.index)
    slippage[is_friday] = SLIPPAGE_BPS / 10000  # 0.02% on rebal days

    targeted_ret = scalar_series * ew_ret - slippage
    return targeted_ret.dropna()


def compute_qqq_returns(closes: pd.DataFrame) -> pd.Series:
    """QQQ buy-and-hold daily returns (proxy for Signal Agg A exposure)."""
    return closes["QQQ"].pct_change().dropna()


def portfolio_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute Sharpe, Sortino, total return, max DD, annual vol, etc."""
    if len(returns) < 30:
        return {}

    ann_ret = returns.mean() * TRADING_DAYS
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS)
    sharpe = (ann_ret - RISK_FREE) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (ann_ret - RISK_FREE) / downside if downside > 0 else 0

    cum = (1 + returns).cumprod()
    total_return = float(cum.iloc[-1] - 1)
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = float(drawdown.min())

    return {
        "label": label,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return_pct": round(total_return * 100, 2),
        "annual_return_pct": round(ann_ret * 100, 2),
        "annual_vol_pct": round(ann_vol * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
    }


def qqq_correlation(returns: pd.Series, qqq_returns: pd.Series) -> float:
    """Correlation with QQQ."""
    aligned = pd.concat([returns, qqq_returns], axis=1).dropna()
    if len(aligned) < 30:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def rebalanced_portfolio(qqq_ret: pd.Series, div_ret: pd.Series,
                          qqq_weight: float, rebal_freq: str) -> pd.Series:
    """
    Simulate portfolio with periodic rebalancing between QQQ and diversifier.
    Drift between rebalance dates.
    """
    aligned = pd.concat([qqq_ret, div_ret], axis=1).dropna()
    aligned.columns = ["qqq", "div"]

    div_weight = 1.0 - qqq_weight

    if qqq_weight == 1.0:
        return aligned["qqq"]
    if qqq_weight == 0.0:
        return aligned["div"]

    # Simple approach: rebalance to target weights at interval, drift between
    n = len(aligned)
    port_returns = pd.Series(0.0, index=aligned.index)

    w_qqq = qqq_weight
    w_div = div_weight

    if rebal_freq == "monthly":
        rebal_months = aligned.index.to_series().dt.to_period("M")
    else:  # quarterly
        rebal_months = aligned.index.to_series().dt.to_period("Q")

    prev_period = None

    for i in range(n):
        date = aligned.index[i]
        r_q = aligned.iloc[i, 0]
        r_d = aligned.iloc[i, 1]

        # Portfolio return for this day with current weights
        port_returns.iloc[i] = w_qqq * r_q + w_div * r_d

        # Update weights for drift
        val_qqq = w_qqq * (1 + r_q)
        val_div = w_div * (1 + r_d)
        total = val_qqq + val_div
        if total > 0:
            w_qqq = val_qqq / total
            w_div = val_div / total

        # Check if we should rebalance
        if rebal_freq == "monthly":
            cur_period = rebal_months.iloc[i]
        else:
            cur_period = rebal_months.iloc[i]

        if prev_period is not None and cur_period != prev_period:
            # Rebalance at period boundary
            w_qqq = qqq_weight
            w_div = div_weight

        prev_period = cur_period

    return port_returns


def estimate_min_account_size():
    """
    Estimate minimum account size where splitting into 3 ETFs makes sense.
    Robinhood: $0 commissions, fractional shares supported.
    Main cost: bid-ask spread on rebalance.
    """
    # Robinhood supports fractional shares for all 3 ETFs
    # No commissions. Main friction is bid-ask spread.
    # GLD spread ~0.01%, UUP spread ~0.03%, QQQ spread ~0.01%
    # Weekly rebalance of GLD+UUP = ~52 * 2 * 0.02% = ~2.08% annual friction
    # But we already account for slippage in the vol-targeted returns

    # Real constraint: minimum meaningful position
    # With fractional shares on Robinhood, even $1 positions work
    # So the question is: is the diversification benefit worth the complexity?

    # Diversification benefit = reduction in max drawdown * account size
    # If MaxDD improves by 5% with diversifier, benefit on $500 = $25
    # Time cost of managing 3 ETFs vs 1: negligible with automation

    # Answer: with fractional shares and $0 commissions, even $500 benefits
    # The real threshold is psychological — is the user willing to manage 3 positions?
    return {
        "minimum_practical": 500,
        "minimum_meaningful_diversification": 1000,
        "reasoning": (
            "Robinhood supports fractional shares with $0 commissions. "
            "There is no hard minimum. At $500+, the diversification benefit "
            "(reduced drawdown) exceeds the minor bid-ask spread costs. "
            "At $1000+, the dollar-value of drawdown reduction becomes meaningful."
        )
    }


def run_backtest():
    """Main backtest."""
    closes = download_data()

    # Compute strategy returns
    qqq_ret = compute_qqq_returns(closes)
    div_ret = compute_gld_uup_vol_targeted(closes)

    print(f"\n--- Individual Strategy Metrics ---")
    qqq_metrics = portfolio_metrics(qqq_ret, "QQQ Buy-Hold (Signal A proxy)")
    div_metrics = portfolio_metrics(div_ret, "GLD+UUP Vol-Targeted 8%")
    qqq_corr_to_qqq = qqq_correlation(div_ret, qqq_ret)

    for m in [qqq_metrics, div_metrics]:
        print(f"  {m['label']}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
              f"Return={m['total_return_pct']:.1f}%, MaxDD={m['max_drawdown_pct']:.1f}%, "
              f"Vol={m['annual_vol_pct']:.1f}%")
    print(f"  Diversifier QQQ correlation: {qqq_corr_to_qqq:.3f}")

    # Test all allocation splits with different rebalancing
    results = {}

    for rebal in REBAL_FREQS:
        print(f"\n--- {rebal.upper()} Rebalancing ---")
        split_results = []

        for qqq_pct_int in range(0, 101, 10):
            qqq_w = qqq_pct_int / 100
            div_w = 1.0 - qqq_w
            label = f"{qqq_pct_int}/{100 - qqq_pct_int}"

            port_ret = rebalanced_portfolio(qqq_ret, div_ret, qqq_w, rebal)
            metrics = portfolio_metrics(port_ret, label)
            metrics["qqq_weight_pct"] = qqq_pct_int
            metrics["div_weight_pct"] = 100 - qqq_pct_int
            metrics["qqq_correlation"] = round(qqq_correlation(port_ret, qqq_ret), 3)
            metrics["rebalance_freq"] = rebal

            split_results.append(metrics)
            print(f"  {label}: Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
                  f"Return={metrics['total_return_pct']:.1f}%, MaxDD={metrics['max_drawdown_pct']:.1f}%, "
                  f"QQQ_corr={metrics['qqq_correlation']:.2f}")

        results[rebal] = split_results

    # Find optimal allocations
    best_sharpe = {}
    best_sortino = {}
    min_maxdd = {}

    for rebal in REBAL_FREQS:
        sr = results[rebal]
        best_sharpe[rebal] = max(sr, key=lambda x: x["sharpe"])
        best_sortino[rebal] = max(sr, key=lambda x: x["sortino"])
        min_maxdd[rebal] = max(sr, key=lambda x: x["max_drawdown_pct"])  # least negative = max

    print("\n" + "=" * 70)
    print("OPTIMAL ALLOCATIONS")
    print("=" * 70)

    for rebal in REBAL_FREQS:
        print(f"\n{rebal.upper()} REBALANCING:")
        bs = best_sharpe[rebal]
        print(f"  Best Sharpe:  {bs['label']} → Sharpe={bs['sharpe']}, "
              f"Return={bs['total_return_pct']:.1f}%, MaxDD={bs['max_drawdown_pct']:.1f}%")
        bso = best_sortino[rebal]
        print(f"  Best Sortino: {bso['label']} → Sortino={bso['sortino']}, "
              f"Return={bso['total_return_pct']:.1f}%, MaxDD={bso['max_drawdown_pct']:.1f}%")
        mm = min_maxdd[rebal]
        print(f"  Min MaxDD:    {mm['label']} → MaxDD={mm['max_drawdown_pct']:.1f}%, "
              f"Sharpe={mm['sharpe']}, Return={mm['total_return_pct']:.1f}%")

    # Account size recommendations
    min_size = estimate_min_account_size()

    account_sizes = [667, 2500, 5000, 10000]
    recommendations = {}

    # Use monthly rebalancing results for recommendations
    monthly_results = results["monthly"]
    best_monthly = best_sharpe["monthly"]

    for acct in account_sizes:
        opt_qqq = best_monthly["qqq_weight_pct"]
        opt_div = best_monthly["div_weight_pct"]

        qqq_dollars = round(acct * opt_qqq / 100, 2)
        gld_dollars = round(acct * opt_div / 100 * 0.5, 2)
        uup_dollars = round(acct * opt_div / 100 * 0.5, 2)

        recommendations[f"${acct:,}"] = {
            "optimal_split": f"{opt_qqq}/{opt_div} (QQQ/Diversifier)",
            "qqq_dollars": qqq_dollars,
            "gld_dollars": gld_dollars,
            "uup_dollars": uup_dollars,
            "expected_sharpe": best_monthly["sharpe"],
            "expected_max_dd_pct": best_monthly["max_drawdown_pct"],
            "note": (
                "Use full diversified allocation" if acct >= 1000
                else "Diversification still helps but dollar amounts are small"
            ),
        }

    print("\n" + "=" * 70)
    print("ACCOUNT SIZE RECOMMENDATIONS")
    print("=" * 70)
    for acct_label, rec in recommendations.items():
        print(f"\n  {acct_label} account:")
        print(f"    Split: {rec['optimal_split']}")
        print(f"    QQQ: ${rec['qqq_dollars']}, GLD: ${rec['gld_dollars']}, UUP: ${rec['uup_dollars']}")
        print(f"    Expected Sharpe: {rec['expected_sharpe']}, MaxDD: {rec['expected_max_dd_pct']:.1f}%")

    # Build output
    output = {
        "generated_at": datetime.now().isoformat(),
        "data_range": f"{START} to {END}",
        "trading_days": len(qqq_ret),
        "individual_strategies": {
            "qqq_buy_hold": qqq_metrics,
            "gld_uup_vol_targeted": {**div_metrics, "qqq_correlation": qqq_corr_to_qqq},
        },
        "allocation_results": {
            rebal: results[rebal] for rebal in REBAL_FREQS
        },
        "optimal_allocations": {
            rebal: {
                "best_sharpe": best_sharpe[rebal],
                "best_sortino": best_sortino[rebal],
                "min_max_drawdown": min_maxdd[rebal],
            }
            for rebal in REBAL_FREQS
        },
        "minimum_account_size": min_size,
        "account_recommendations": recommendations,
    }

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")

    return output


if __name__ == "__main__":
    run_backtest()
