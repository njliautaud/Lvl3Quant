#!/usr/bin/env python3
"""
Weekly/Monthly Timeframe Strategy Backtest
==========================================
6 variants targeting uncorrelated-to-QQQ returns on weekly/monthly horizons.
OOT: 2022-01-01 to 2026-07-29, capital $645, slippage 0.02%, commission $0.

Variants:
  A: Monthly Momentum Top-1 (CTA style)
  B: Weekly GLD/TLT Switching
  C: Monthly Macro Regime (Quarterly Lookback)
  D: Weekly Risk Parity (GLD+TLT+UUP)
  E: Monthly Dual Momentum (Absolute + Relative)
  F: Weekly Contrarian (Buy Last Week's Loser)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DATA_START = "2020-01-01"  # extra history for lookback
TICKERS = ["SPY", "QQQ", "GLD", "TLT", "UUP", "IWM", "EEM", "RSP", "DBA", "SHY"]
VIX_TICKER = "^VIX"
PERM_ITERS = 1000
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/weekly_monthly_timeframe_results.json"


def download_data():
    """Download all required data from yfinance."""
    all_tickers = TICKERS + [VIX_TICKER]
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    close = data["Close"]
    # Ensure columns are strings
    close.columns = [str(c) for c in close.columns]
    close = close.ffill()
    print(f"  Data shape: {close.shape}, range: {close.index[0].date()} to {close.index[-1].date()}")
    return close


def apply_slippage(returns, slippage=SLIPPAGE_PCT):
    """Apply slippage on rebalance days (where allocation changes)."""
    return returns  # slippage applied per-strategy at trade points


def calc_metrics(equity_curve, qqq_returns, name, trades_count, capital=CAPITAL):
    """Calculate performance metrics for a strategy."""
    returns = equity_curve.pct_change().dropna()
    if len(returns) < 2:
        return None

    # Align with QQQ
    common_idx = returns.index.intersection(qqq_returns.index)
    if len(common_idx) < 20:
        return None
    ret_aligned = returns.loc[common_idx]
    qqq_aligned = qqq_returns.loc[common_idx]

    # Basic metrics
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    ann_factor = 252
    ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(ann_factor) if len(downside) > 0 else 1e-9
    sortino = ann_return / downside_vol

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # QQQ correlation
    qqq_corr = ret_aligned.corr(qqq_aligned)

    # Profit factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Win rate (daily)
    wr = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    return {
        "name": name,
        "total_return_pct": round(total_return * 100, 2),
        "ann_return_pct": round(ann_return * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "qqq_correlation": round(qqq_corr, 4),
        "trades": trades_count,
        "final_equity": round(equity_curve.iloc[-1], 2),
        "start_equity": round(equity_curve.iloc[0], 2),
    }


def permutation_test(strategy_returns, n_iter=PERM_ITERS):
    """Permutation test for Sharpe ratio significance."""
    if len(strategy_returns) < 20:
        return 1.0
    observed_sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    count_ge = 0
    rng = np.random.RandomState(42)
    ret_vals = strategy_returns.values.copy()
    for _ in range(n_iter):
        shuffled = ret_vals.copy()
        # Random sign flip (permutation of direction)
        signs = rng.choice([-1, 1], size=len(shuffled))
        shuffled = shuffled * signs
        shuf_sharpe = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
        if shuf_sharpe >= observed_sharpe:
            count_ge += 1
    return round(count_ge / n_iter, 4)


def regime_split(equity_curve, spy_close):
    """Split returns into bull/bear regimes based on SPY 200-SMA."""
    spy_sma200 = spy_close.rolling(200).mean()
    returns = equity_curve.pct_change().dropna()

    common = returns.index.intersection(spy_sma200.dropna().index).intersection(spy_close.index)
    ret_c = returns.loc[common]
    spy_c = spy_close.loc[common]
    sma_c = spy_sma200.loc[common]

    bull_mask = spy_c > sma_c
    bear_mask = ~bull_mask

    bull_ret = ret_c[bull_mask]
    bear_ret = ret_c[bear_mask]

    def sharpe_from_ret(r):
        if len(r) < 5:
            return 0.0
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    bull_sharpe = sharpe_from_ret(bull_ret)
    bear_sharpe = sharpe_from_ret(bear_ret)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "bull_days": int(bull_mask.sum()),
        "bear_days": int(bear_mask.sum()),
        "regime_gap": round(regime_gap, 4),
    }


def get_weekly_fridays(idx, start, end):
    """Get Friday dates within range from index."""
    mask = (idx >= start) & (idx <= end)
    subset = idx[mask]
    # Group by week, take last trading day
    weekly = subset.to_series().groupby(pd.Grouper(freq="W-FRI")).last().dropna()
    return pd.DatetimeIndex(weekly.values)


def get_monthly_ends(idx, start, end):
    """Get last trading day of each month within range."""
    mask = (idx >= start) & (idx <= end)
    subset = idx[mask]
    monthly = subset.to_series().groupby(pd.Grouper(freq="ME")).last().dropna()
    return pd.DatetimeIndex(monthly.values)


def get_quarterly_ends(idx, start, end):
    """Get last trading day of each quarter within range."""
    mask = (idx >= start) & (idx <= end)
    subset = idx[mask]
    quarterly = subset.to_series().groupby(pd.Grouper(freq="QE")).last().dropna()
    return pd.DatetimeIndex(quarterly.values)


# ── Strategy Implementations ───────────────────────────────────────────────

def strategy_a_monthly_momentum(close, oot_start, oot_end):
    """A: Monthly Momentum Top-1 (CTA style)
    Rank GLD, TLT, UUP, DBA, EEM by 12m return (skip last 1m). Long best. Monthly rebal.
    """
    assets = ["GLD", "TLT", "UUP", "DBA", "EEM"]
    rebal_dates = get_monthly_ends(close.index, oot_start, oot_end)

    equity = CAPITAL
    equity_series = {}
    trades = 0
    current_asset = None

    for i, date in enumerate(rebal_dates):
        # 12m return excluding last 1m
        end_lookback = date - pd.DateOffset(months=1)
        start_lookback = date - pd.DateOffset(months=12)

        returns_12m = {}
        for a in assets:
            prices = close[a].loc[:end_lookback]
            prices_start = close[a].loc[:start_lookback]
            if len(prices) > 20 and len(prices_start) > 20:
                ret = prices.iloc[-1] / prices_start.iloc[-1] - 1
                returns_12m[a] = ret

        if not returns_12m:
            # Fill daily equity for this month
            if i + 1 < len(rebal_dates):
                next_date = rebal_dates[i + 1]
            else:
                next_date = pd.Timestamp(oot_end)
            days = close.index[(close.index > date) & (close.index <= next_date)]
            for d in days:
                equity_series[d] = equity
            continue

        best = max(returns_12m, key=returns_12m.get)

        if best != current_asset:
            # Apply slippage on switch
            equity *= (1 - SLIPPAGE_PCT)
            trades += 1
            current_asset = best

        # Hold until next rebal
        if i + 1 < len(rebal_dates):
            next_date = rebal_dates[i + 1]
        else:
            next_date = pd.Timestamp(oot_end)

        hold_days = close.index[(close.index > date) & (close.index <= next_date)]
        if len(hold_days) > 0 and date in close.index:
            entry_price = close[current_asset].loc[date]
            for d in hold_days:
                price = close[current_asset].loc[d]
                equity_series[d] = equity * (price / entry_price)
            equity = equity_series[hold_days[-1]]

    eq = pd.Series(equity_series).sort_index()
    return eq, trades


def strategy_b_weekly_gld_tlt(close, oot_start, oot_end):
    """B: Weekly GLD/TLT Switching. If GLD 4w return > TLT 4w → long GLD, else TLT."""
    rebal_dates = get_weekly_fridays(close.index, oot_start, oot_end)

    equity = CAPITAL
    equity_series = {}
    trades = 0
    current_asset = None

    for i, date in enumerate(rebal_dates):
        start_lookback = date - pd.DateOffset(weeks=4)
        gld_prices = close["GLD"].loc[start_lookback:date]
        tlt_prices = close["TLT"].loc[start_lookback:date]

        if len(gld_prices) < 5 or len(tlt_prices) < 5:
            continue

        gld_ret = gld_prices.iloc[-1] / gld_prices.iloc[0] - 1
        tlt_ret = tlt_prices.iloc[-1] / tlt_prices.iloc[0] - 1

        chosen = "GLD" if gld_ret > tlt_ret else "TLT"

        if chosen != current_asset:
            equity *= (1 - SLIPPAGE_PCT)
            trades += 1
            current_asset = chosen

        if i + 1 < len(rebal_dates):
            next_date = rebal_dates[i + 1]
        else:
            next_date = pd.Timestamp(oot_end)

        hold_days = close.index[(close.index > date) & (close.index <= next_date)]
        if len(hold_days) > 0 and date in close.index:
            entry_price = close[current_asset].loc[date]
            for d in hold_days:
                price = close[current_asset].loc[d]
                equity_series[d] = equity * (price / entry_price)
            equity = equity_series[hold_days[-1]] if len(hold_days) > 0 else equity

    eq = pd.Series(equity_series).sort_index()
    return eq, trades


def strategy_c_macro_regime(close, oot_start, oot_end):
    """C: Monthly Macro Regime (Quarterly Lookback).
    SPY 3m > 5% → RSP; VIX avg > 25 → GLD; else UUP. Quarterly rebal.
    """
    vix = close["^VIX"] if "^VIX" in close.columns else None
    rebal_dates = get_quarterly_ends(close.index, oot_start, oot_end)

    equity = CAPITAL
    equity_series = {}
    trades = 0
    current_asset = None

    for i, date in enumerate(rebal_dates):
        start_q = date - pd.DateOffset(months=3)

        spy_prices = close["SPY"].loc[start_q:date]
        spy_3m_ret = (spy_prices.iloc[-1] / spy_prices.iloc[0] - 1) if len(spy_prices) > 5 else 0

        vix_avg = 20
        if vix is not None:
            vix_q = vix.loc[start_q:date]
            if len(vix_q) > 5:
                vix_avg = vix_q.mean()

        if spy_3m_ret > 0.05:
            chosen = "RSP"
        elif vix_avg > 25:
            chosen = "GLD"
        else:
            chosen = "UUP"

        if chosen != current_asset:
            equity *= (1 - SLIPPAGE_PCT)
            trades += 1
            current_asset = chosen

        if i + 1 < len(rebal_dates):
            next_date = rebal_dates[i + 1]
        else:
            next_date = pd.Timestamp(oot_end)

        hold_days = close.index[(close.index > date) & (close.index <= next_date)]
        if len(hold_days) > 0 and date in close.index:
            entry_price = close[current_asset].loc[date]
            for d in hold_days:
                price = close[current_asset].loc[d]
                equity_series[d] = equity * (price / entry_price)
            equity = equity_series[hold_days[-1]] if len(hold_days) > 0 else equity

    eq = pd.Series(equity_series).sort_index()
    return eq, trades


def strategy_d_risk_parity(close, oot_start, oot_end):
    """D: Weekly Risk Parity (GLD+TLT+UUP).
    Allocate inversely proportional to 4-week volatility. Rebalance Fridays.
    """
    assets = ["GLD", "TLT", "UUP"]
    rebal_dates = get_weekly_fridays(close.index, oot_start, oot_end)

    equity = CAPITAL
    equity_series = {}
    trades = 0
    prev_weights = None

    for i, date in enumerate(rebal_dates):
        start_lookback = date - pd.DateOffset(weeks=4)

        vols = {}
        for a in assets:
            prices = close[a].loc[start_lookback:date]
            if len(prices) > 5:
                ret = prices.pct_change().dropna()
                vols[a] = ret.std() if ret.std() > 0 else 1e-9
            else:
                vols[a] = 1e-9

        inv_vol = {a: 1.0 / v for a, v in vols.items()}
        total_inv = sum(inv_vol.values())
        weights = {a: inv_vol[a] / total_inv for a in assets}

        # Check if weights changed materially
        if prev_weights is None or any(abs(weights[a] - prev_weights.get(a, 0)) > 0.05 for a in assets):
            equity *= (1 - SLIPPAGE_PCT)
            trades += 1
        prev_weights = weights

        if i + 1 < len(rebal_dates):
            next_date = rebal_dates[i + 1]
        else:
            next_date = pd.Timestamp(oot_end)

        hold_days = close.index[(close.index > date) & (close.index <= next_date)]
        if len(hold_days) > 0 and date in close.index:
            entry_prices = {a: close[a].loc[date] for a in assets}
            for d in hold_days:
                port_val = 0
                for a in assets:
                    price = close[a].loc[d]
                    port_val += equity * weights[a] * (price / entry_prices[a])
                equity_series[d] = port_val
            equity = equity_series[hold_days[-1]] if len(hold_days) > 0 else equity

    eq = pd.Series(equity_series).sort_index()
    return eq, trades


def strategy_e_dual_momentum(close, oot_start, oot_end):
    """E: Monthly Dual Momentum (Absolute + Relative) for GLD and TLT.
    Both positive → long higher 12m return. One positive → long it. Neither → SHY.
    """
    rebal_dates = get_monthly_ends(close.index, oot_start, oot_end)

    equity = CAPITAL
    equity_series = {}
    trades = 0
    current_asset = None

    for i, date in enumerate(rebal_dates):
        start_12m = date - pd.DateOffset(months=12)

        gld_12m = 0
        tlt_12m = 0
        gld_start = close["GLD"].loc[:start_12m]
        tlt_start = close["TLT"].loc[:start_12m]

        if len(gld_start) > 0:
            gld_12m = close["GLD"].loc[date] / gld_start.iloc[-1] - 1
        if len(tlt_start) > 0:
            tlt_12m = close["TLT"].loc[date] / tlt_start.iloc[-1] - 1

        gld_pos = gld_12m > 0
        tlt_pos = tlt_12m > 0

        if gld_pos and tlt_pos:
            chosen = "GLD" if gld_12m > tlt_12m else "TLT"
        elif gld_pos:
            chosen = "GLD"
        elif tlt_pos:
            chosen = "TLT"
        else:
            chosen = "SHY"

        if chosen != current_asset:
            equity *= (1 - SLIPPAGE_PCT)
            trades += 1
            current_asset = chosen

        if i + 1 < len(rebal_dates):
            next_date = rebal_dates[i + 1]
        else:
            next_date = pd.Timestamp(oot_end)

        hold_days = close.index[(close.index > date) & (close.index <= next_date)]
        if len(hold_days) > 0 and date in close.index:
            entry_price = close[current_asset].loc[date]
            for d in hold_days:
                price = close[current_asset].loc[d]
                equity_series[d] = equity * (price / entry_price)
            equity = equity_series[hold_days[-1]] if len(hold_days) > 0 else equity

    eq = pd.Series(equity_series).sort_index()
    return eq, trades


def strategy_f_weekly_contrarian(close, oot_start, oot_end):
    """F: Weekly Contrarian. Long last week's worst performer among GLD, TLT, UUP, DBA."""
    assets = ["GLD", "TLT", "UUP", "DBA"]
    rebal_dates = get_weekly_fridays(close.index, oot_start, oot_end)

    equity = CAPITAL
    equity_series = {}
    trades = 0
    current_asset = None

    for i, date in enumerate(rebal_dates):
        start_1w = date - pd.DateOffset(weeks=1)

        week_rets = {}
        for a in assets:
            prices = close[a].loc[start_1w:date]
            if len(prices) >= 2:
                week_rets[a] = prices.iloc[-1] / prices.iloc[0] - 1

        if not week_rets:
            continue

        worst = min(week_rets, key=week_rets.get)

        if worst != current_asset:
            equity *= (1 - SLIPPAGE_PCT)
            trades += 1
            current_asset = worst

        if i + 1 < len(rebal_dates):
            next_date = rebal_dates[i + 1]
        else:
            next_date = pd.Timestamp(oot_end)

        hold_days = close.index[(close.index > date) & (close.index <= next_date)]
        if len(hold_days) > 0 and date in close.index:
            entry_price = close[current_asset].loc[date]
            for d in hold_days:
                price = close[current_asset].loc[d]
                equity_series[d] = equity * (price / entry_price)
            equity = equity_series[hold_days[-1]] if len(hold_days) > 0 else equity

    eq = pd.Series(equity_series).sort_index()
    return eq, trades


def validate_strategy(eq, qqq_returns, spy_close, name, trades):
    """Run 5-gate validation on a strategy."""
    metrics = calc_metrics(eq, qqq_returns, name, trades)
    if metrics is None:
        return {"name": name, "status": "INSUFFICIENT_DATA", "gates": {}}

    returns = eq.pct_change().dropna()

    # Gate 1: Sharpe > 0.5
    gate1 = metrics["sharpe"] > 0.5

    # Gate 2: Permutation p < 0.05
    perm_p = permutation_test(returns)
    gate2 = perm_p < 0.05

    # Gate 3: Regime gap < 0.5
    regime = regime_split(eq, spy_close)
    gate3 = regime["regime_gap"] < 0.5

    # Gate 4: MDD > -50%
    gate4 = metrics["max_drawdown_pct"] > -50

    # Gate 5: Trades >= 20
    gate5 = trades >= 20

    gates_passed = sum([gate1, gate2, gate3, gate4, gate5])

    return {
        **metrics,
        "perm_p_value": perm_p,
        "regime": regime,
        "gates": {
            "sharpe_gt_0.5": gate1,
            "perm_p_lt_0.05": gate2,
            "regime_gap_lt_0.5": gate3,
            "mdd_gt_neg50": gate4,
            "trades_gte_20": gate5,
        },
        "gates_passed": f"{gates_passed}/5",
        "status": "PASS" if gates_passed == 5 else "FAIL",
    }


def main():
    print("=" * 70)
    print("WEEKLY/MONTHLY TIMEFRAME STRATEGY BACKTEST")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${CAPITAL}")
    print("=" * 70)

    close = download_data()

    # QQQ benchmark returns
    qqq_oot = close["QQQ"].loc[OOT_START:OOT_END]
    qqq_returns = qqq_oot.pct_change().dropna()
    spy_close = close["SPY"].loc[OOT_START:OOT_END]

    strategies = [
        ("A: Monthly Momentum Top-1", strategy_a_monthly_momentum),
        ("B: Weekly GLD/TLT Switching", strategy_b_weekly_gld_tlt),
        ("C: Quarterly Macro Regime", strategy_c_macro_regime),
        ("D: Weekly Risk Parity", strategy_d_risk_parity),
        ("E: Monthly Dual Momentum", strategy_e_dual_momentum),
        ("F: Weekly Contrarian", strategy_f_weekly_contrarian),
    ]

    results = []
    for name, func in strategies:
        print(f"\nRunning {name}...")
        eq, trades = func(close, OOT_START, OOT_END)
        if len(eq) < 20:
            print(f"  SKIP: insufficient data ({len(eq)} points)")
            results.append({"name": name, "status": "INSUFFICIENT_DATA"})
            continue

        validation = validate_strategy(eq, qqq_returns, spy_close, name, trades)
        results.append(validation)

        print(f"  Sharpe: {validation.get('sharpe', 'N/A')}, "
              f"Return: {validation.get('total_return_pct', 'N/A')}%, "
              f"MDD: {validation.get('max_drawdown_pct', 'N/A')}%, "
              f"QQQ corr: {validation.get('qqq_correlation', 'N/A')}, "
              f"Trades: {trades}, "
              f"Gates: {validation.get('gates_passed', 'N/A')}, "
              f"Status: {validation.get('status', 'N/A')}")

    # QQQ benchmark
    qqq_eq = qqq_oot / qqq_oot.iloc[0] * CAPITAL
    qqq_metrics = calc_metrics(qqq_eq, qqq_returns, "QQQ Benchmark (buy-hold)", 1)

    output = {
        "backtest_meta": {
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "commission": 0,
            "permutation_iterations": PERM_ITERS,
            "run_timestamp": datetime.now().isoformat(),
        },
        "qqq_benchmark": qqq_metrics,
        "strategies": results,
        "summary": {
            "total_variants": len(results),
            "passed_all_gates": sum(1 for r in results if r.get("status") == "PASS"),
            "best_sharpe": max((r.get("sharpe", -999) for r in results if r.get("sharpe")), default=None),
            "lowest_qqq_corr": min((abs(r.get("qqq_correlation", 999)) for r in results if r.get("qqq_correlation") is not None), default=None),
        },
    }

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"QQQ Benchmark: Sharpe {qqq_metrics['sharpe']}, Return {qqq_metrics['total_return_pct']}%")
    print(f"\nVariants passing all 5 gates: {output['summary']['passed_all_gates']}/{len(results)}")
    print(f"Best Sharpe: {output['summary']['best_sharpe']}")
    print(f"Lowest |QQQ corr|: {output['summary']['lowest_qqq_corr']}")

    print("\n{:<35} {:>7} {:>8} {:>8} {:>8} {:>6} {:>7}".format(
        "Strategy", "Sharpe", "Return%", "MDD%", "QQQcorr", "Trd", "Gates"))
    print("-" * 90)
    for r in results:
        if r.get("status") == "INSUFFICIENT_DATA":
            print(f"{r['name']:<35} {'INSUFFICIENT DATA':>50}")
            continue
        print("{:<35} {:>7.3f} {:>8.1f} {:>8.1f} {:>8.4f} {:>6d} {:>7}".format(
            r["name"], r.get("sharpe", 0), r.get("total_return_pct", 0),
            r.get("max_drawdown_pct", 0), r.get("qqq_correlation", 0),
            r.get("trades", 0), r.get("gates_passed", "N/A")))

    # Save results
    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return output


if __name__ == "__main__":
    main()
